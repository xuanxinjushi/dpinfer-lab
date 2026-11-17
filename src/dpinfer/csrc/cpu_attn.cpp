// Bandwidth-oriented CPU decode attention for GQA with bf16 / fp16 KV (AVX2 + FMA + F16C).
//
// q   [Hkv, G, D] fp32      (the G query heads sharing one KV head)
// k,v [Hkv, L, D] bf16/fp16 (contiguous)
// returns out [Hkv, G, D] fp32 (normalised), lse [Hkv, G] fp32 (natural log, includes scale)
//
// Split-KV: work items = (kv head, chunk of CHUNK tokens), run with OpenMP. Inside a chunk:
// pass 1 reads K once and computes G scores per token, pass 2 reads V once and accumulates
// p * v for the G queries. Partials (m, l, u) are merged per head with log-sum-exp.
// D must be a multiple of 8 and <= 256.
#include <torch/extension.h>
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <vector>

namespace {

constexpr int CHUNK = 256;
constexpr int MAXG = 8;
constexpr int MAXD = 256;

template <bool BF16>
inline __m256 load8(const uint16_t* p) {
  __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  if constexpr (BF16) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
  } else {
    return _mm256_cvtph_ps(h);
  }
}

inline float hsum(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

template <bool BF16>
void chunk_partial(const float* q, const uint16_t* k, const uint16_t* v, int G, int D, int n,
                   float scale, float* m_out, float* l_out, float* u_out) {
  float s[MAXG][CHUNK];
  // pass 1: scores
  for (int j = 0; j < n; ++j) {
    const uint16_t* kj = k + (size_t)j * D;
    __m256 acc[MAXG];
    for (int g = 0; g < G; ++g) acc[g] = _mm256_setzero_ps();
    for (int d = 0; d < D; d += 8) {
      __m256 kv = load8<BF16>(kj + d);
      for (int g = 0; g < G; ++g) acc[g] = _mm256_fmadd_ps(_mm256_loadu_ps(q + g * D + d), kv, acc[g]);
    }
    for (int g = 0; g < G; ++g) s[g][j] = hsum(acc[g]) * scale;
  }
  // softmax pieces
  for (int g = 0; g < G; ++g) {
    float m = -INFINITY;
    for (int j = 0; j < n; ++j) m = std::max(m, s[g][j]);
    float l = 0.f;
    for (int j = 0; j < n; ++j) { s[g][j] = std::exp(s[g][j] - m); l += s[g][j]; }
    m_out[g] = m; l_out[g] = l;
  }
  // pass 2: u = sum p * v
  __m256 u[MAXG][MAXD / 8];
  for (int g = 0; g < G; ++g)
    for (int c = 0; c < D / 8; ++c) u[g][c] = _mm256_setzero_ps();
  for (int j = 0; j < n; ++j) {
    const uint16_t* vj = v + (size_t)j * D;
    for (int c = 0; c < D / 8; ++c) {
      __m256 vv = load8<BF16>(vj + c * 8);
      for (int g = 0; g < G; ++g) u[g][c] = _mm256_fmadd_ps(_mm256_set1_ps(s[g][j]), vv, u[g][c]);
    }
  }
  for (int g = 0; g < G; ++g)
    for (int c = 0; c < D / 8; ++c) _mm256_storeu_ps(u_out + g * D + c * 8, u[g][c]);
}

}  // namespace

std::vector<torch::Tensor> gqa_decode(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                      double scale, int64_t threads) {
  TORCH_CHECK(q.dtype() == torch::kFloat32 && q.is_contiguous());
  TORCH_CHECK(k.is_contiguous() && v.is_contiguous() && k.dtype() == v.dtype());
  const bool bf16 = k.dtype() == torch::kBFloat16;
  TORCH_CHECK(bf16 || k.dtype() == torch::kFloat16, "k/v must be bf16 or fp16");
  const int H = k.size(0), L = k.size(1), D = k.size(2), G = q.size(1);
  TORCH_CHECK(q.size(0) == H && q.size(2) == D && D % 8 == 0 && D <= MAXD && G <= MAXG);

  const int nch = (L + CHUNK - 1) / CHUNK;
  auto pm = torch::empty({H, nch, G}, torch::kFloat32);
  auto pl = torch::empty({H, nch, G}, torch::kFloat32);
  auto pu = torch::empty({H, nch, G, D}, torch::kFloat32);
  const float* qp = q.data_ptr<float>();
  const uint16_t* kp = reinterpret_cast<const uint16_t*>(k.data_ptr());
  const uint16_t* vp = reinterpret_cast<const uint16_t*>(v.data_ptr());
  float *mp = pm.data_ptr<float>(), *lp = pl.data_ptr<float>(), *up = pu.data_ptr<float>();
  const float sc = (float)scale;

#pragma omp parallel for schedule(static) num_threads(threads)
  for (int w = 0; w < H * nch; ++w) {
    int h = w / nch, c = w % nch;
    int j0 = c * CHUNK, n = std::min(CHUNK, L - j0);
    const uint16_t* kc = kp + ((size_t)h * L + j0) * D;
    const uint16_t* vc = vp + ((size_t)h * L + j0) * D;
    const float* qh = qp + (size_t)h * G * D;
    size_t o = (size_t)h * nch + c;
    if (bf16) chunk_partial<true>(qh, kc, vc, G, D, n, sc, mp + o * G, lp + o * G, up + o * G * D);
    else      chunk_partial<false>(qh, kc, vc, G, D, n, sc, mp + o * G, lp + o * G, up + o * G * D);
  }

  // merge chunks per head
  auto out = torch::zeros({H, G, D}, torch::kFloat32);
  auto lse = torch::empty({H, G}, torch::kFloat32);
  float *op = out.data_ptr<float>(), *lsep = lse.data_ptr<float>();
  for (int h = 0; h < H; ++h)
    for (int g = 0; g < G; ++g) {
      float M = -INFINITY;
      for (int c = 0; c < nch; ++c) M = std::max(M, mp[((size_t)h * nch + c) * G + g]);
      float Ls = 0.f;
      float* oh = op + ((size_t)h * G + g) * D;
      for (int c = 0; c < nch; ++c) {
        size_t o = ((size_t)h * nch + c) * G + g;
        float w = std::exp(mp[o] - M);
        Ls += w * lp[o];
        const float* uc = up + o * D;
        for (int d = 0; d < D; ++d) oh[d] += w * uc[d];
      }
      for (int d = 0; d < D; ++d) oh[d] /= Ls;
      lsep[h * G + g] = M + std::log(Ls);
    }
  return {out, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gqa_decode", &gqa_decode, "GQA decode attention, bf16/fp16 KV, AVX2");
}
