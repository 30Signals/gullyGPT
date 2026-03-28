# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

gullyGPT treats T20 cricket matches as a ball-by-ball sequence generation problem. Qwen2.5-3B is fine-tuned via LoRA on structured ball sequences from Cricsheet data, then used to autoregressively simulate full matches in a 2-player Streamlit game.

## Key Commands

```bash
# Data pipeline (Phase 1) — run locally
python src/data/filter.py                        # produces data/processed/t20_index.jsonl
python src/data/serialize.py                     # produces train.txt / val.txt
python src/data/serialize.py --sample 5          # preview 5 sequences

# Training (Phase 2) — GPU server only
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python src/train/train.py --config src/train/config.yaml > train.log 2>&1 &
# Resume from checkpoint:
python src/train/train.py --config src/train/config.yaml --resume-from checkpoints/qwen-cricket/checkpoint-500

# Validate model (Phase 3)
python src/generate/eval.py --checkpoint checkpoints/qwen-cricket --n 10
python src/generate/eval.py --checkpoint checkpoints/qwen-cricket --n 3 --verbose

# Headless simulation (single match)
python src/generate/engine.py --checkpoint checkpoints/qwen-cricket \
  --team1 India --team2 Pakistan --venue Wankhede --toss-winner India --toss-decision bat

# Run the app (Phase 4)
streamlit run src/app/game.py

# GPU server utilities
bash scripts/watch_training.sh       # live training monitor (polls every 30s)
bash scripts/pull_checkpoint.sh      # scp latest checkpoint from GPU server
```

## Architecture

**Phase 1 — Data Pipeline** (`src/data/`)
- `filter.py` — scans `data/all_json/` (21k Cricsheet JSONs), keeps T20/IT20, writes `data/processed/t20_index.jsonl`
- `serialize.py` — converts each match JSON into a flat tagged string (see format below), writes `data/processed/train.txt` and `val.txt` (90/10 split by match, not by ball)
- `dataset.py` — `MatchDataset`: one sample per match (~12k samples), tokenized and truncated to `max_seq_len`. `__getitem__` returns only `{"input_ids": list}` — no `labels` key; `DataCollatorForLanguageModeling` handles labels

**Phase 2 — Training** (`src/train/`) — GPU server only
- `train.py` — loads Qwen2.5-3B in bfloat16, applies LoRA (r=16, targets q/k/v/o_proj), `gradient_checkpointing=False` (re-enabling it interacts badly with LoRA). `lr` must be cast via `float(cfg["lr"])` — YAML parses `2e-4` as a string
- `config.yaml` — `batch_size: 2, grad_accumulation: 16` (effective batch=32). Fits in ~19GB on L4 at ~13s/step

**Phase 3 — Generation Engine** (`src/generate/`)
- `engine.py` — `MatchEngine` maintains `_match_header` + `_ball_lines` list (rolling 60-ball context window). `generate_ball()` builds a prefix hint, stops generation at newline (not `</ball>` — that's multi-token), appends `</ball>` if missing. Engine enforces hard rules (20 overs, 10 wickets, 6 legal balls/over); model handles outcomes
- `commentary.py` — calls `LLMRouter` with a Harsha Bhogle-style prompt, returns one sentence per ball. Reads config from `llm_router_config.json` in repo root. Silent no-op on any error
- `llm_router.py` — vendored from `30Signals/autoresearch`. Round-robin across OpenRouter/Gemini/Groq slots with per-slot exponential backoff, powered by litellm
- `eval.py` — generates N full matches headlessly, validates scorecards (wickets ≤ 10, overs ≤ 20, sane scores), prints aggregate stats

**Phase 4 — App** (`src/app/game.py`)
- Streamlit 2-player UI. Player 1 (batting) picks openers at setup and next batter after each wicket. Player 2 (bowling) picks bowler before each over. All match state in `st.session_state`. Commentary renders as an italicised blockquote after each ball

## Ball Sequence Format

```
<match> format=T20 venue=Wankhede teams=MI,CSK toss=MI:bat season=2023 </match>
<inn1> batting=MI </inn1>
<ball> ov=0.1 bwl=Bumrah bat=Rohit nst=Ishan runs=0 </ball>
<ball> ov=0.3 bwl=Bumrah bat=Rohit nst=Ishan runs=0 wicket=caught out=Rohit fld=Dhoni </ball>
<ball> ov=0.4 bwl=Bumrah bat=Kohli nst=Ishan extras=wide runs=1 </ball>
<inn2> batting=CSK </inn2>
...
<result> winner=MI by=23runs </result>
```

Names use underscores internally (spaces replaced); `parse_ball_str()` in `engine.py` reverses this on output.

## Configuration

- `.env` — gitignored; copy from `.env.example` and fill in values. Holds GPU server connection details (`GPU_HOST`, `GPU_KEY`, `GPU_REMOTE_DIR`) and LLM API keys (`OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, etc.)
- `llm_router_config.json` — gitignored; copy from `llm_router_config.json.example`. Uses `${ENV_VAR}` placeholders that are expanded from `.env` at runtime by `commentary.py`
- `.gpu_server` — legacy, superseded by `.env`

## Data

- Raw: `data/all_json/` — 21,379 Cricsheet JSONs (not committed)
- Processed: `data/processed/` — 13,427 T20 matches, ~9M tokens total, ~500–700 tokens/match (not committed)

## Dependencies

```bash
pip install transformers>=4.40 peft>=0.10 torch>=2.2 datasets accelerate streamlit tqdm pyyaml litellm>=1.40.0
```
