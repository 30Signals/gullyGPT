# gullyGPT

> **Game of Cricket is just next-token prediction.**

---

## The Idea

gullyGPT is an experiment inspired by Andrej Karpathy's insight that **intelligence emerges from simple next-token prediction**.

Instead of building statistics or hardcoding rules, this project treats cricket matches as a ball-by-ball **sequence generation problem**:

- Every delivery is a token
- A match is a sequence
- A model that truly understands cricket must learn to predict what happens next — dot ball, boundary, wicket, wide — given everything that came before

If a language model can predict the next word in a sentence because it has internalized grammar, semantics, and world knowledge, then a model trained on ball-by-ball cricket data should internalize the game itself: pitch conditions, bowler form, batter tendencies, match pressure, momentum.

No hand-crafted features. No Duckworth-Lewis lookup tables. Just: **given this sequence of deliveries, what comes next?**

---

## Why This is Interesting

Cricket is uniquely rich for this kind of experiment:

- **Long context dependencies** — a batter's form in over 3 matters in over 17
- **Discrete, structured tokens** — each ball has a small, well-defined outcome space
- **Hidden state** — fatigue, confidence, match situation encode themselves into the sequence
- **Strategy emerges** — bowling changes, batting aggression, and death-over tactics are all implicit in the data

If the model learns to predict well, it has necessarily learned cricket.

---

## How It Works

### Data

13,100 T20 matches from [Cricsheet](https://cricsheet.org) serialized into a tagged sequence format:

```
<match> format=T20 venue=Wankhede teams=India,Pakistan toss=India:bat season=2023 </match>
<inn1> batting=India </inn1>
<ball> ov=0.1 bwl=Bumrah bat=Rohit nst=Gill runs=0 </ball>
<ball> ov=0.2 bwl=Bumrah bat=Rohit nst=Gill runs=4 </ball>
<ball> ov=0.3 bwl=Bumrah bat=Rohit nst=Gill runs=0 wicket=caught out=Rohit fld=Rizwan </ball>
...
<result> winner=India by=23runs </result>
```

Each match is ~500–700 tokens. ~9M tokens total across the corpus.

### Model

**Qwen2.5-3B** fine-tuned via **LoRA** (r=16, targeting q/k/v/o projections) on the ball sequence corpus. Causal LM objective: predict the next token in the sequence. The model learns run distributions, wicket probabilities, and match dynamics purely from data.

Training: ~3 epochs on an NVIDIA L4 GPU, ~4 hours.

### Generation Engine

`MatchEngine` generates one ball at a time autoregressively:

1. Build context from match header + rolling window of last 60 balls
2. Feed into model with a prefix hint (`<ball> ov=5.3 bwl=Bumrah bat=Kohli ...`)
3. Sample until newline
4. Parse the generated ball string back into structured data
5. Enforce hard cricket rules (max 20 overs, 10 wickets, 6 legal balls/over)

### Commentary

Each generated ball is passed to an **LLM router** (backed by Gemini/Groq/OpenRouter) which produces one line of punchy live commentary in the voice of Harsha Bhogle.

### App

A **2-player Streamlit app** where:
- **Player 1** (batting) picks openers and selects the next batter after each wicket
- **Player 2** (bowling) picks the bowler for each over

The model generates every delivery. Commentary appears after each ball.

---

## Project Structure

```
src/
├── data/
│   ├── filter.py       # extract T20 matches from Cricsheet JSONs
│   ├── serialize.py    # convert match → tagged ball sequence
│   └── dataset.py      # HuggingFace Dataset wrapper (MatchDataset)
├── train/
│   ├── train.py        # LoRA fine-tune on GPU server
│   └── config.yaml     # hyperparams
├── generate/
│   ├── engine.py       # MatchEngine — autoregressive ball generation
│   ├── eval.py         # headless match simulation + scorecard validation
│   ├── commentary.py   # per-ball commentary via LLM router
│   └── llm_router.py   # multi-provider LLM router (vendored)
└── app/
    └── game.py         # Streamlit 2-player UI
```

---

## Running It

### Prerequisites

```bash
pip install -r requirements.txt
cp llm_router_config.json.example llm_router_config.json
# fill in your API keys (OpenRouter / Gemini / Groq — all have free tiers)
```

### Play

```bash
streamlit run src/app/game.py
```

Requires a fine-tuned checkpoint at `checkpoints/qwen-cricket` (or specify path in the UI).

### Validate the model

```bash
python src/generate/eval.py --checkpoint checkpoints/qwen-cricket --n 10
```

### Headless simulation

```bash
python src/generate/engine.py \
    --checkpoint checkpoints/qwen-cricket \
    --team1 India --team2 Pakistan \
    --venue Wankhede --toss-winner India --toss-decision bat
```

---

## Status

Training complete. Commentary layer live. Streamlit app playable.

---

## Inspiration

- [Andrej Karpathy — The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- The beautiful, chaotic game of cricket
