# gullyGPT — Implementation Plan

## Context

Cricket as next-token prediction. Instead of rules or stats, we fine-tune Qwen 3B on ball-by-ball T20 sequences so the model learns match dynamics purely from prediction. End goal: a 2-player interactive match simulator where the model generates deliveries autoregressively.

**Decisions locked in:**
- Level 2 (structured simulator), not raw commentary
- T20 only to start (13k+ matches, fixed 20-over structure)
- Fine-tune Qwen 3B via LoRA — faster results, still principled
- GPU training on user's separate server

---

## Project Structure

```
gullygpt/
├── data/
│   ├── all_json/           # raw cricsheet (21k files, already downloaded)
│   └── processed/          # output of pipeline
│       ├── t20_sequences.jsonl     # one match per line
│       └── train.txt / val.txt    # final training corpus
├── src/
│   ├── data/
│   │   ├── filter.py       # extract T20 matches, build metadata index
│   │   ├── serialize.py    # convert match → sequence of ball strings
│   │   └── dataset.py      # HuggingFace Dataset wrapper
│   ├── train/
│   │   ├── train.py        # LoRA fine-tune Qwen 3B (runs on GPU server)
│   │   └── config.yaml     # hyperparams, paths
│   ├── generate/
│   │   └── engine.py       # autoregressive match generation, stopping logic
│   └── app/
│       └── game.py         # Streamlit 2-player UI
├── requirements.txt
└── CLAUDE.md
```

---

## Phase 1 — Data Pipeline

### 1a. Filter & index T20 matches (`src/data/filter.py`)

Read all 21k JSONs, keep `match_type == "T20"` (and optionally IT20).
Output: `data/processed/t20_index.jsonl` — one line per match with path + metadata.

### 1b. Sequence serialization (`src/data/serialize.py`)

Convert each match into a flat string sequence. Format:

```
<match> format=T20 venue=Wankhede teams=MI,CSK toss=MI:bat season=2023 </match>
<inn1> batting=MI </inn1>
<ball> ov=0.1 bwl=Bumrah bat=Rohit nst=Ishan runs=0 </ball>
<ball> ov=0.2 bwl=Bumrah bat=Rohit nst=Ishan runs=4 </ball>
<ball> ov=0.3 bwl=Bumrah bat=Rohit nst=Ishan runs=0 wicket=caught out=Rohit fld=Dhoni </ball>
<ball> ov=0.4 bwl=Bumrah bat=Kohli nst=Ishan extras=wide runs=1 </ball>
...
<inn2> batting=CSK </inn2>
...
<result> winner=MI by=23runs </result>
```

Why this format:
- Human-readable → Qwen's tokenizer handles it well
- Structured enough for reliable parsing during inference
- Short tokens keep sequence length manageable (~500-700 tokens per T20 match)

Edge cases to handle: extras (wide, noball, bye, legbye), multiple wickets in a ball, DRS reviews (skip), replacements (skip).

Output: `data/processed/train.txt` and `val.txt` (90/10 split, split by match not by ball).

### 1c. Dataset wrapper (`src/data/dataset.py`)

HuggingFace `Dataset` that tokenizes sequences for causal LM training. Packs multiple short matches into context windows of 2048 tokens.

---

## Phase 2 — Fine-tuning Qwen 3B

### Model: `Qwen/Qwen2.5-3B` (or `Qwen2.5-3B-Instruct`)

### Training script (`src/train/train.py`)

- HuggingFace `transformers` + `peft` (LoRA)
- LoRA config: `r=16, alpha=32, target_modules=["q_proj","v_proj"]`
- Causal LM objective: predict next token in ball sequence
- ~3M ball tokens across 13k matches — expect 2-3 epochs
- Runs on user's GPU server; outputs checkpoint to `checkpoints/`

### Config (`src/train/config.yaml`)

```yaml
model: Qwen/Qwen2.5-3B
lora_r: 16
lora_alpha: 32
batch_size: 8
grad_accumulation: 4
lr: 2e-4
epochs: 3
max_seq_len: 2048
data_path: data/processed/
output_dir: checkpoints/
```

### Evaluation

Perplexity on val set. Also: manually generate 5 full T20 matches and sanity-check that run totals, wicket counts, over counts are valid.

---

## Phase 3 — Generation Engine

### `src/generate/engine.py`

`MatchEngine` class:
- Takes: `team1`, `team2`, `venue`, `toss_winner`, `toss_decision`, `pitch`
- Maintains running context string (appends each generated ball)
- Generates one ball at a time: feeds context → model → sample next tokens until `</ball>` tag
- Parses the generated ball string back into a dict
- Enforces hard cricket rules (max 6 legal balls per over, max 10 wickets, 20 overs) as stopping conditions — model handles *what* happens, engine handles *when to stop*

Sampling: temperature=0.8, top-p=0.9 for variety. Lower temp for high-pressure situations (last over, 9 wickets down) — configurable.

---

## Phase 4 — 2-Player Streamlit App

### `src/app/game.py`

**Setup screen:**
- Player 1 (batting): select team, pick 2 openers from squad
- Player 2 (bowling): select team, pick opening bowler
- Select venue, pitch type (flat / green / dusty / slow)

**Match loop:**
- Engine generates ball → display outcome card
- After each over: Player 2 picks next bowler (dropdown)
- After wicket: Player 1 picks next batter
- Scoreboard updates live: runs, wickets, RRR, CRR
- Between innings: show scorecard, target set

**Commentary (optional, Phase 4b):**
- Pass generated ball dict to a second prompt on Qwen/Claude to produce one line of commentary
- Displayed below each ball card

**UI stack:** Streamlit (fast to ship), `st.session_state` for match state.

---

## Build Order

1. `src/data/filter.py` + `src/data/serialize.py` — run locally, produces training corpus
2. `src/data/dataset.py` — local
3. `src/train/train.py` + `config.yaml` — ship to GPU server, train
4. `src/generate/engine.py` — local, test with checkpoint
5. `src/app/game.py` — local Streamlit

---

## Verification

- After Phase 1: `python src/data/serialize.py --sample 5` — eyeball 5 match sequences
- After Phase 2: eval perplexity < 3.0 on val; generate 3 matches, check no rule violations
- After Phase 3: run `engine.py` headless, generate 10 full matches, assert valid scorecards
- After Phase 4: full playthrough of one match in Streamlit end-to-end

---

## Dependencies (requirements.txt)

```
transformers>=4.40
peft>=0.10
torch>=2.2
datasets
accelerate
streamlit
tqdm
```
