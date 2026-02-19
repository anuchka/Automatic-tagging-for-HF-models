#!/usr/bin/env python3
"""
Tag HuggingFace models: rule-based from HF metadata, then AI supplementation.

Hybrid flow:
  1. Fetch model metadata (and optionally README) from HuggingFace public API.
  2. Apply rule-based tags from metadata (param count, context length, keywords).
  3. For "rule-only" tags (size_*, long-context, desktop-deployable) we keep
     rule results only (we have numeric data).
  4. For semantic tags (chat, reasoning, tool-calling, etc.), we merge rule
     results with AI: send metadata + README snippet to an LLM and ask which
     tags apply, then take the union.
  5. Final tags = rule-only from rules + (rule ∪ AI) for the rest.

Usage: python hf_model_tagger.py <model_url_or_id> [--no-ai]
Example: python hf_model_tagger.py https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
"""

import argparse
import json
import os
import re
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ---------------------------------------------------------------------------
# TAG DEFINITIONS — add/remove/edit here to change behavior
# ---------------------------------------------------------------------------
# Each entry: tag_id -> (description, condition_func)
# condition_func(data) receives model_data dict, returns True if tag applies.

def _make_tag_def(description: str, condition):
    """Helper so we only store (description, condition) per tag."""
    return (description, condition)

def _text_contains(data: dict, *keywords: str) -> bool:
    """True if any of model id, tags, or pipeline contain any keyword (case-insensitive)."""
    text_parts = []
    text_parts.append(data.get("modelId") or data.get("id") or "")
    text_parts.append(data.get("pipeline_tag") or "")
    for t in data.get("tags") or []:
        text_parts.append(t if isinstance(t, str) else "")
    card = data.get("cardData") or {}
    for t in card.get("tags") or []:
        text_parts.append(t if isinstance(t, str) else "")
    combined = " ".join(text_parts).lower()
    return any(kw.lower() in combined for kw in keywords)

def _get_param_count(data: dict) -> float | None:
    """Return parameter count in billions, or None if unknown."""
    safetensors = data.get("safetensors") or {}
    total = safetensors.get("total")
    if total is not None:
        return total / 1e9
    # Some repos expose total under parameters (e.g. parameters.BF16 = param count)
    params = safetensors.get("parameters")
    if isinstance(params, dict):
        for v in params.values():
            if isinstance(v, (int, float)):
                return v / 1e9
            break
    elif isinstance(params, (int, float)):
        return params / 1e9
    config = data.get("config") or {}
    n = config.get("num_parameters")
    if n is not None:
        return n / 1e9
    import re
    name = (data.get("id") or "").lower().split("/")[-1]
    m = re.search(r"(\d+\.?\d*)b$", name)
    if m:
        return float(m.group(1))
    return None

def _get_max_position(data: dict) -> int | None:
    """Max context length from config (tokens)."""
    config = data.get("config") or {}
    return config.get("max_position_embeddings")

# Tag conditions (easy to add/remove; order doesn't affect output set)
TAG_DEFINITIONS = {
    "tool-calling": _make_tag_def(
        "Model supports function/API calling",
        lambda d: _text_contains(d, "tool", "function calling", "function_calling", "api calling"),
    ),
    "chat": _make_tag_def(
        "Model is fine-tuned for conversation",
        lambda d: _text_contains(d, "chat", "instruct", "conversational", "conversation"),
    ),
    "code-generation": _make_tag_def(
        "Model is trained primarily on code",
        lambda d: _text_contains(d, "code", "codellama", "starcoder", "codegen", "code-generation"),
    ),
    "instruction-following": _make_tag_def(
        "Model is trained to follow instructions",
        lambda d: _text_contains(d, "instruct", "instruction", "alpaca", "vicuna"),
    ),
    "reasoning": _make_tag_def(
        "Model does multi-step logical or math reasoning",
        lambda d: _text_contains(d, "reasoning", "cot", "chain-of-thought", "math", "gsm", "orca"),
    ),
    "long-context": _make_tag_def(
        "Context window is 100K tokens or more",
        lambda d: (_get_max_position(d) or 0) >= 100_000,
    ),
    "size_tiny": _make_tag_def(
        "Less than 1B parameters",
        lambda d: (_get_param_count(d) or 0) > 0 and (_get_param_count(d) or 0) < 1,
    ),
    "size_small": _make_tag_def(
        "1–3B parameters",
        lambda d: (v := _get_param_count(d)) is not None and 1 <= v < 3,
    ),
    "size_medium": _make_tag_def(
        "3–8B parameters",
        lambda d: (v := _get_param_count(d)) is not None and 3 <= v <= 8,
    ),
    "size_large": _make_tag_def(
        "8–15B parameters",
        lambda d: (v := _get_param_count(d)) is not None and 8 < v <= 15,
    ),
    "base-model": _make_tag_def(
        "Not instruction-tuned or fine-tuned for chat",
        lambda d: not _text_contains(d, "instruct", "chat", "conversational", "finetuned", "fine-tuned"),
    ),
    "guardrail": _make_tag_def(
        "Designed for safety or content moderation",
        lambda d: _text_contains(d, "guardrail", "safety", "moderation", "content moderation"),
    ),
    "embedding": _make_tag_def(
        "Produces vector representations for search/RAG",
        lambda d: _text_contains(d, "embedding", "sentence-transformers", "rag", "retrieval") or (d.get("pipeline_tag") or "").lower() == "sentence-similarity",
    ),
    "desktop-deployable": _make_tag_def(
        "Can run on a standard laptop (16GB RAM or less)",
        lambda d: (v := _get_param_count(d)) is not None and v <= 13,
    ),
}

# Tags we never let AI override — we have exact numeric data from HF.
RULE_ONLY_TAGS = {
    "size_tiny", "size_small", "size_medium", "size_large",
    "long-context", "desktop-deployable",
}

# ---------------------------------------------------------------------------
# API (no auth)
# ---------------------------------------------------------------------------
HF_API_BASE = "https://huggingface.co/api/models"
HF_RAW_BASE = "https://huggingface.co"

def parse_model_id(url_or_id: str) -> str:
    """Extract repo_id from URL or return as-is if already 'org/name'."""
    s = url_or_id.strip()
    # URL forms: https://huggingface.co/org/model-name  or  org/model-name
    m = re.match(r"https?://(?:www\.)?huggingface\.co/([^/]+/[^/?\s]+)", s)
    if m:
        return m.group(1).rstrip("/")
    if "/" in s and " " not in s:
        return s
    raise ValueError(f"Could not parse model ID from: {url_or_id}")

def fetch_model_data(repo_id: str) -> dict:
    """Fetch model metadata from HuggingFace public API (no token)."""
    url = f"{HF_API_BASE}/{repo_id}"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=15) as resp:
            return json.load(resp)
    except HTTPError as e:
        if e.code == 404:
            raise SystemExit(f"Model not found: {repo_id}")
        raise SystemExit(f"HTTP {e.code}: {url}")
    except URLError as e:
        raise SystemExit(f"Network error: {e.reason}")

def fetch_config(repo_id: str) -> dict:
    """Fetch config.json for max_position_embeddings and param count if needed."""
    url = f"{HF_RAW_BASE}/{repo_id}/raw/main/config.json"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            return json.load(resp)
    except (HTTPError, URLError):
        return {}

def fetch_readme(repo_id: str, max_chars: int = 4000) -> str:
    """Fetch README.md for model card context (for AI). Returns first max_chars."""
    url = f"{HF_RAW_BASE}/{repo_id}/raw/main/README.md"
    req = Request(url, headers={"Accept": "text/plain"})
    try:
        with urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return raw[:max_chars].strip()
    except (HTTPError, URLError):
        return ""

def apply_tags(model_data: dict) -> list[str]:
    """Apply all tag rules; return sorted list of tag ids that apply."""
    applied = []
    for tag_id, (_, condition) in TAG_DEFINITIONS.items():
        try:
            if condition(model_data):
                applied.append(tag_id)
        except Exception:
            continue
    return sorted(applied)

def merge_rule_and_ai_tags(rule_tags: list[str], ai_tags: list[str]) -> list[str]:
    """Hybrid merge: keep rule-only from rules; for the rest take union of rule + AI."""
    rule_set = set(rule_tags)
    ai_set = {t for t in ai_tags if t in TAG_DEFINITIONS}
    # Rule-only: only from rules (we have numeric data).
    rule_only_result = rule_set & RULE_ONLY_TAGS
    # Semantic: union of rule + AI (AI supplements).
    semantic_tag_ids = set(TAG_DEFINITIONS) - RULE_ONLY_TAGS
    semantic_result = (rule_set | ai_set) & semantic_tag_ids
    return sorted(rule_only_result | semantic_result)

# ---------------------------------------------------------------------------
# AI supplementation (OpenAI or Anthropic)
# ---------------------------------------------------------------------------
def _build_model_summary(model_data: dict, readme_snippet: str) -> str:
    """Summary of model for the LLM."""
    parts = [
        f"Model ID: {model_data.get('modelId') or model_data.get('id') or 'unknown'}",
        f"Pipeline: {model_data.get('pipeline_tag') or 'unknown'}",
        f"Tags: {', '.join(model_data.get('tags') or [])}",
    ]
    card = model_data.get("cardData") or {}
    if card.get("tags"):
        parts.append(f"Card tags: {', '.join(card['tags'])}")
    config = model_data.get("config") or {}
    if config.get("max_position_embeddings"):
        parts.append(f"Max position embeddings: {config['max_position_embeddings']}")
    if readme_snippet:
        parts.append("\nModel card (excerpt):\n" + readme_snippet)
    return "\n".join(parts)

def _build_tag_list_for_prompt() -> str:
    """Tag IDs and descriptions for the prompt (semantic only, so AI doesn't guess size)."""
    lines = []
    for tag_id in sorted(TAG_DEFINITIONS):
        if tag_id in RULE_ONLY_TAGS:
            continue
        desc = TAG_DEFINITIONS[tag_id][0]
        lines.append(f"  - {tag_id}: {desc}")
    return "\n".join(lines)

def _call_openai(prompt: str, api_key: str) -> str:
    """Call OpenAI Chat Completions; return content or raise."""
    url = "https://api.openai.com/v1/chat/completions"
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0,
    }
    req = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    choice = (data.get("choices") or [None])[0]
    if not choice:
        raise RuntimeError("OpenAI returned no choices")
    msg = choice.get("message") or {}
    return (msg.get("content") or "").strip()

def _call_anthropic(prompt: str, api_key: str) -> str:
    """Call Anthropic Messages API; return content or raise."""
    url = "https://api.anthropic.com/v1/messages"
    body = {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    for block in data.get("content") or []:
        if block.get("type") == "text":
            return (block.get("text") or "").strip()
    raise RuntimeError("Anthropic returned no text content")

def _parse_ai_tag_list(content: str) -> list[str]:
    """Extract tag IDs from LLM response (one per line, or comma-separated)."""
    valid = set(TAG_DEFINITIONS)
    found = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Drop bullets and numbering
        line = re.sub(r"^[\s\-*]\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        for part in re.split(r"[,;]", line):
            tag = part.strip().lower().replace(" ", "_")
            if tag in valid:
                found.append(tag)
    return list(dict.fromkeys(found))  # dedupe, keep order

def ask_ai_for_tags(model_data: dict, readme_snippet: str, verbose: bool = False) -> list[str]:
    """
    Ask an LLM which tags apply. Uses OPENAI_API_KEY or ANTHROPIC_API_KEY.
    Returns list of tag IDs. Returns [] if no key or on error.
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not openai_key and not anthropic_key:
        if verbose:
            print("No OPENAI_API_KEY or ANTHROPIC_API_KEY set; skipping AI.", file=sys.stderr)
        return []

    summary = _build_model_summary(model_data, readme_snippet)
    tag_list = _build_tag_list_for_prompt()
    prompt = f"""You are classifying a HuggingFace model. Given the metadata below, which of these tags apply? Reply with only the tag IDs that apply, one per line. Use only the exact tag IDs listed — no other text.

Tags to choose from:
{tag_list}

Model metadata:
{summary}

Tag IDs (one per line):"""

    try:
        if openai_key:
            content = _call_openai(prompt, openai_key)
        else:
            content = _call_anthropic(prompt, anthropic_key)
        return _parse_ai_tag_list(content)
    except Exception as e:
        if verbose:
            print(f"AI call failed: {e}", file=sys.stderr)
        return []

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Tag a HuggingFace model from its URL using the public API (no login).",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="HuggingFace model URL or repo ID (e.g. mistralai/Mistral-7B-Instruct-v0.2)",
    )
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="List all available tag IDs and descriptions, then exit.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only print the list of tags (one per line).",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Skip AI supplementation; use only rule-based tags from metadata.",
    )
    args = parser.parse_args()

    if args.list_tags:
        print("Available tags:")
        for tag_id, (desc, _) in sorted(TAG_DEFINITIONS.items()):
            print(f"  {tag_id}: {desc}")
        return

    if not args.model:
        parser.error("the following arguments are required: model (or use --list-tags / --test)")

    repo_id = parse_model_id(args.model)
    if not args.quiet:
        print(f"Fetching metadata for {repo_id} ...", file=sys.stderr)

    model_data = fetch_model_data(repo_id)
    # Merge in config if not already present (API sometimes omits full config)
    if not model_data.get("config") or "max_position_embeddings" not in (model_data.get("config") or {}):
        config = fetch_config(repo_id)
        if config:
            model_data["config"] = {**(model_data.get("config") or {}), **config}

    rule_tags = apply_tags(model_data)
    tags = list(rule_tags)
    ai_tags: list[str] = []

    if False:
        if not args.quiet:
            print("Fetching README for AI context ...", file=sys.stderr)
        readme_snippet = fetch_readme(repo_id)
        if not args.quiet:
            print("Asking AI for tag supplementation ...", file=sys.stderr)
        ai_tags = ask_ai_for_tags(model_data, readme_snippet, verbose=not args.quiet)
        tags = merge_rule_and_ai_tags(rule_tags, ai_tags)

    if args.quiet:
        for t in tags:
            print(t)
    else:
        print(f"Model: {model_data.get('modelId') or repo_id}")
        print(f"Tags ({len(tags)}): {', '.join(tags)}")
        for t in tags:
            if t in TAG_DEFINITIONS:
                src = " (AI)" if ai_tags and t in ai_tags and t not in RULE_ONLY_TAGS else ""
                print(f"  - {t}: {TAG_DEFINITIONS[t][0]}{src}")

def run_offline_test():
    """Test tagging logic with canned Mistral-7B-Instruct-v0.2 API response (no network)."""
    # Minimal copy of the real API response for mistralai/Mistral-7B-Instruct-v0.2
    model_data = {
        "id": "mistralai/Mistral-7B-Instruct-v0.2",
        "modelId": "mistralai/Mistral-7B-Instruct-v0.2",
        "pipeline_tag": "text-generation",
        "tags": ["transformers", "pytorch", "text-generation", "finetuned", "conversational", "mistral"],
        "config": {"max_position_embeddings": 32768},
        "cardData": {"tags": ["finetuned", "mistral-common"]},
        "safetensors": {"total": 7_241_732_096},
    }
    tags = apply_tags(model_data)
    expected = {"chat", "instruction-following", "size_medium", "desktop-deployable"}
    got = set(tags)
    assert expected <= got, f"Expected at least {expected}, got {got}"
    print("Offline test passed. Tags:", sorted(tags))

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--test":
        run_offline_test()
        sys.exit(0)
    main()
