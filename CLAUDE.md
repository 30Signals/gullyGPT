# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

gullyGPT treats T20 cricket matches as a ball-by-ball sequence generation problem. A Qwen 3B model is fine-tuned via LoRA on structured ball sequences from Cricsheet data, then used to autoregressively simulate full matches in a 2-player Streamlit game. See `.claude/plan.md` for the full implementation plan.

## Architecture

The project has 4 phases, each building on the last:

**Phase 1 — Data Pipeline** (`src/data/`)
- `filter.py` — scans `data/all_json/` (21k Cricsheet JSONs), keeps T20/IT20 matches, writes `data/processed/t20_index.jsonl`
- `serialize.py` — converts each match JSON into a flat tagged string sequence (see format below), writes `data/processed/train.txt` and `val.txt`
- `dataset.py` — HuggingFace `Dataset` wrapper that tokenizes and packs sequences into 2048-token windows for causal LM training

**Phase 2 — Training** (`src/train/`) — runs on GPU server, not locally
- `train.py` — LoRA fine-tune of `Qwen/Qwen2.5-3B` using HuggingFace `transformers` + `peft`
- `config.yaml` — all hyperparams (model path, LoRA rank, batch size, epochs, output dir)

**Phase 3 — Generation Engine** (`src/generate/engine.py`)
- `MatchEngine` class generates one ball at a time: feeds running context → model → samples until `</ball>` tag, parses result dict
- Hard cricket rules (max 20 overs, 10 wickets, 6 legal balls/over) enforced by the engine; the model handles *what* happens

**Phase 4 — App** (`src/app/game.py`)
- Streamlit 2-player UI: Player 1 (batting) picks batters after wickets, Player 2 (bowling) picks bowlers after each over
- All match state in `st.session_state`

## Ball Sequence Format

Every match serializes to this tagged string — this is the vocabulary the model learns:

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

## Key Commands

```bash
# Run data pipeline (Phase 1)
python src/data/filter.py
python src/data/serialize.py
python src/data/serialize.py --sample 5   # preview 5 sequences

# Training — run on GPU server after syncing data/processed/
python src/train/train.py --config src/train/config.yaml

# Run the game locally (needs checkpoint)
streamlit run src/app/game.py
```

## Data

- Raw data: `data/all_json/` — 21,379 Cricsheet match JSONs (already downloaded, not committed to git)
- T20 matches: ~13,100 of the 21,379 (61%)
- Each T20 match serializes to ~500–700 tokens

## Dependencies

```bash
pip install transformers>=4.40 peft>=0.10 torch>=2.2 datasets accelerate streamlit tqdm
```
