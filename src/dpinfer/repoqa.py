"""Build fixed-length RepoQA 'search needle function' prompts from real repository code.

Files are concatenated in dependency order (dependencies first), each with a path header.
A window of code around the needle function is cut so that the full chat-templated prompt
is close to (never above) the requested token count, snapped to whole lines.
"""
import gzip
import json
from pathlib import Path

DATASET = Path(__file__).resolve().parents[2] / "data/repoqa_release/repoqa-2024-06-23.json.gz"

SYSTEM = "You are a helpful assistant good at code understanding."
INSTRUCTION = (
    "Based on the function description and code context, please retrieve and repeat "
    "the exact described function from the code context in a code block wrapped by ```:"
)


def load_repos(path=DATASET):
    data = json.load(gzip.open(path))
    return {r["repo"]: dict(r, lang=lang) for lang, repos in data.items() for r in repos}


def ordered_files(repo):
    """Topological order over repo['dependency'] (deps before dependents), stable by path."""
    deps = repo["dependency"]
    order, seen = [], set()

    def visit(p, stack):
        if p in seen or p in stack or p not in repo["content"]:
            return
        stack.add(p)
        for q in sorted(deps.get(p, [])):
            if q != p:
                visit(q, stack)
        stack.discard(p)
        seen.add(p)
        order.append(p)

    for p in sorted(repo["content"]):
        visit(p, set())
    return order


def _concat(repo):
    parts, offsets, pos = [], {}, 0
    for p in ordered_files(repo):
        header = f"# ===== {p} =====\n"
        body = repo["content"][p]
        offsets[p] = pos + len(header)
        chunk = header + body + ("" if body.endswith("\n") else "\n")
        parts.append(chunk)
        pos += len(chunk)
    return "".join(parts), offsets


def _byte_to_char(text, b):
    return len(text.encode("utf-8")[:b].decode("utf-8", errors="ignore"))


def messages_for(code, description):
    user = (
        f"{INSTRUCTION}\n\nDescription:\n{description}\n\n"
        f"Code context:\n```\n{code}```\n\n"
        f"{INSTRUCTION}\n\nDescription:\n{description}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def prompt_len(tokenizer, messages):
    return len(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_dict=True)["input_ids"])


def build(tokenizer, repo, needle_idx, target_tokens, depth=0.5):
    """Return (messages, info). `depth` = where the needle sits in the code window (0=start, 1=end)."""
    needle = repo["needles"][needle_idx]
    text, offsets = _concat(repo)
    fbody = repo["content"][needle["path"]]
    n_cs = offsets[needle["path"]] + _byte_to_char(fbody, needle["start_byte"])
    n_ce = offsets[needle["path"]] + _byte_to_char(fbody, needle["end_byte"])

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    n_tok = len(offs)
    ts = next(i for i, (a, b) in enumerate(offs) if b > n_cs)
    te = next(i for i, (a, b) in enumerate(offs) if b >= n_ce) + 1

    overhead = prompt_len(tokenizer, messages_for("", needle["description"]))
    budget = target_tokens - overhead
    if te - ts > budget:
        raise ValueError(f"needle ({te - ts} tok) longer than code budget ({budget})")
    if budget > n_tok:
        raise ValueError(f"repo has {n_tok} tokens < code budget {budget}")

    def window(budget):
        start = ts - int((budget - (te - ts)) * depth)
        start = max(0, min(start, n_tok - budget))
        cs, ce = offs[start][0], offs[start + budget - 1][1]
        if start > 0:  # snap to whole lines
            cs = text.index("\n", cs) + 1 if "\n" in text[cs:n_cs] else cs
        if ce < len(text):
            nl = text.rfind("\n", n_ce, ce)
            ce = nl + 1 if nl != -1 else ce
        return text[cs:ce]

    # Re-tokenisation of the joined prompt can drift by a few tokens; shrink until it fits.
    for _ in range(20):
        code = window(budget)
        msgs = messages_for(code, needle["description"])
        n = prompt_len(tokenizer, msgs)
        if n <= target_tokens:
            break
        budget -= (n - target_tokens) + 8

    info = {
        "repo": repo["repo"], "lang": repo["lang"], "commit_sha": repo["commit_sha"],
        "needle_idx": needle_idx, "needle_name": needle["name"], "needle_path": needle["path"],
        "needle_tokens": te - ts, "depth": depth, "target_tokens": target_tokens,
        "prompt_tokens": n, "code_chars": len(code),
        "files_in_window": sum(1 for p in offsets if f"# ===== {p} =====" in code),
    }
    assert repo["content"][needle["path"]].encode()[needle["start_byte"]:needle["end_byte"]].decode(
        errors="ignore").strip()[:200] in code, "needle not in window"
    return msgs, info
