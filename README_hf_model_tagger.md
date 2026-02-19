# HuggingFace model tagger

Tag HuggingFace models using a **hybrid flow**: rule-based tags from HF metadata, then **AI supplementation** for semantic tags (e.g. reasoning, tool-calling) when an API key is set.

## Hybrid flow

1. **Fetch** model metadata (and optionally README) from the HuggingFace public API (no login).
2. **Rule-based tags** from metadata: param count → size_*, desktop-deployable; context length → long-context; keywords in id/tags → chat, instruction-following, etc.
3. **Rule-only tags** (size_*, long-context, desktop-deployable) are never overridden by AI — we keep the rule result.
4. **AI supplementation**: if `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set and you don’t pass `--no-ai`, the script sends a summary of the model + README excerpt to an LLM and asks which of the *semantic* tags apply. Results are merged (union with rule-based semantic tags).
5. **Final tags** = rule-only from rules + (rule ∪ AI) for the rest.

## Usage

```bash
# With AI supplementation (set OPENAI_API_KEY or ANTHROPIC_API_KEY)
python3 hf_model_tagger.py <model_url_or_id>

# Rules only (no API key needed)
python3 hf_model_tagger.py --no-ai <model_url_or_id>
```

Examples:

```bash
python3 hf_model_tagger.py https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
python3 hf_model_tagger.py --no-ai mistralai/Mistral-7B-Instruct-v0.2
```

Options:

- `--list-tags` — List all tag IDs and descriptions.
- `-q` / `--quiet` — Print only the tag names (one per line).
- `--no-ai` — Skip AI; use only rule-based tags from metadata.
- `--test` — Run offline test (no network).

### AI (optional)

- **OpenAI**: set `OPENAI_API_KEY`. Uses `gpt-4o-mini`.
- **Anthropic**: set `ANTHROPIC_API_KEY`. Uses `claude-3-5-haiku`.
- If both are set, OpenAI is used. If neither is set, only rule-based tags are used (same as `--no-ai`).
- Tags marked `(AI)` in the output were suggested by the LLM (semantic tags only; size/context tags stay rule-based).

## Tags applied

| Tag | When applied |
|-----|----------------|
| `tool-calling` | Model supports function/API calling |
| `chat` | Model is fine-tuned for conversation |
| `code-generation` | Model is trained primarily on code |
| `instruction-following` | Model is trained to follow instructions |
| `reasoning` | Multi-step logical or math reasoning |
| `long-context` | Context window ≥ 100K tokens |
| `size_tiny` | &lt; 1B parameters |
| `size_small` | 1–3B parameters |
| `size_medium` | 3–8B parameters |
| `size_large` | 8–15B parameters |
| `base-model` | Not instruction-tuned or chat-tuned |
| `guardrail` | Safety or content moderation |
| `embedding` | Vector representations for search/RAG |
| `desktop-deployable` | Can run on laptop (≤8B, ~16GB RAM) |

## Adding or removing tags

Edit the `TAG_DEFINITIONS` dict in `hf_model_tagger.py`. Each entry is:

```python
"tag-id": (description_string, condition_function)
```

`condition_function(data)` receives the model API payload and returns `True` if the tag applies. Add new entries or delete existing ones; the script uses only what’s in the dict.

## How to test

**Option A: Run the test script** (in your Terminal, from this directory):

```bash
cd /Users/anna
./test_tagger.sh
```

This runs: (1) offline test, (2) list tags, (3) live rules-only (`--no-ai`), (4) live with AI if `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set.

**Option B: Run steps manually**

1. **Offline test (no network)** — checks rule-based logic:

   ```bash
   python3 hf_model_tagger.py --test
   ```
   Expected: `Offline test passed. Tags: ['chat', 'desktop-deployable', 'instruction-following', 'size_medium']`

2. **Rules only (live, no API key)** — HF metadata only:

   ```bash
   python3 hf_model_tagger.py --no-ai https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
   ```

3. **With AI supplementation** — set a key, then run without `--no-ai`:

   ```bash
   export OPENAI_API_KEY=sk-your-key   # or ANTHROPIC_API_KEY=...
   python3 hf_model_tagger.py https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
   ```
   Tags suggested by the LLM are marked `(AI)` in the output.

4. **List all tag definitions:**

   ```bash
   python3 hf_model_tagger.py --list-tags
   ```

**If `python3` fails** (e.g. Xcode prompt on macOS), use Anaconda:

```bash
/opt/anaconda3/bin/python hf_model_tagger.py --test
/opt/anaconda3/bin/python hf_model_tagger.py --no-ai "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2"
```

## Requirements

- Python 3.9+
- No extra packages (uses only the standard library and public HuggingFace API).
