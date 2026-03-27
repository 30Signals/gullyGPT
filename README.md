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

- **Long context dependencies** — a batter's form in over 3 matters in over 47
- **Discrete, structured tokens** — each ball has a small, well-defined outcome space
- **Hidden state** — fatigue, confidence, match situation encode themselves into the sequence
- **Strategy emerges** — field placements, bowling changes, batting aggression are all implicit in the data

If the model learns to predict well, it has necessarily learned cricket.

---

## Framing

```
match = [b1, b2, b3, ..., bN]

P(bN | b1, b2, ..., bN-1)
```

Each ball `b` encodes: bowler, batter, over, delivery, outcome (runs/wicket/extras), and match context.

The model's job: given the history, predict the next delivery's outcome.

---

## Status

Early experiment. Work in progress.

---

## Inspiration

- [Andrej Karpathy — The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- The beautiful, chaotic game of cricket
