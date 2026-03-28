"""
Commentary layer: takes a Ball + match context and generates one line of
punchy cricket commentary using the LLM router (30Signals/autoresearch).

Reads model config from ~/.llm_router_config.json automatically.

Usage:
    from src.generate.commentary import get_commentary
    line = get_commentary(ball, state)
"""

import sys
from pathlib import Path

# Make llm_router importable regardless of working directory
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from llm_router import LLMRouter  # noqa: E402 (local copy in src/generate/)
from src.generate.engine import Ball, InningsState

_router: LLMRouter | None = None

# Config lives in repo root
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "llm_router_config.json"


def _get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter(config_path=str(_CONFIG_PATH))
    return _router


_SYSTEM = (
    "You are a cricket commentator with the voice of Harsha Bhogle — sharp, warm, and vivid. "
    "Given a single delivery, produce exactly ONE punchy sentence of live commentary. "
    "Rules: under 20 words, no clichés, use the real player names, make wickets dramatic, "
    "sixes soar, and dot balls feel tense. No extra text — just the one sentence."
)


def get_commentary(ball: Ball, state: InningsState, innings_num: int = 1) -> str:
    """Return a single commentary line for the given ball. Falls back to empty string on error."""
    try:
        target_str = ""
        if state.target:
            needed = state.target - state.runs
            balls_left = (20 - state.overs_complete) * 6 - state.legal_balls
            target_str = f" Target: {state.target}, need {needed} off {balls_left} balls."

        situation = (
            f"Innings {innings_num}. "
            f"Score: {state.runs}/{state.wickets}, Over {ball.over}. "
            f"{ball.bowler} bowls to {ball.batter}.{target_str}"
        )

        if ball.is_wicket:
            delivery = f"OUT! {ball.player_out} {ball.wicket}"
            if ball.fielder:
                delivery += f", caught by {ball.fielder}"
        elif ball.extras == "wide":
            delivery = "Wide ball."
        elif ball.extras == "noball":
            delivery = f"No-ball! {ball.runs} run(s)."
        else:
            delivery = f"{ball.runs} run(s)."

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"{situation} Outcome: {delivery}"},
        ]

        router = _get_router()
        response = router.chat(messages, max_tokens=60)
        return response.choices[0].message.content.strip()
    except Exception:
        return ""
