"""
Phase 3: Autoregressive match generation engine.
Generates one ball at a time by feeding the running context into the model
and sampling until </ball> is produced.

Usage (headless test):
    python src/generate/engine.py \
        --checkpoint checkpoints/qwen-cricket \
        --team1 India --team2 Pakistan \
        --venue Wankhede --toss-winner India --toss-decision bat
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Ball:
    over: float
    bowler: str
    batter: str
    non_striker: str
    runs: int
    extras: Optional[str] = None
    wicket: Optional[str] = None
    player_out: Optional[str] = None
    fielder: Optional[str] = None

    @property
    def is_legal(self) -> bool:
        """Wides and no-balls don't count as legal deliveries."""
        return self.extras not in ("wide", "noball")

    @property
    def is_wicket(self) -> bool:
        return self.wicket is not None

    def __str__(self) -> str:
        desc = f"Over {self.over} | {self.bowler} to {self.batter}: {self.runs} run(s)"
        if self.extras:
            desc += f" [{self.extras}]"
        if self.wicket:
            desc += f" | WICKET! {self.player_out} {self.wicket}"
            if self.fielder:
                desc += f" (c {self.fielder})"
        return desc


@dataclass
class InningsState:
    batting_team: str
    bowling_team: str
    runs: int = 0
    wickets: int = 0
    legal_balls: int = 0   # legal deliveries in current over
    overs_complete: int = 0
    total_balls: int = 0
    balls: list = field(default_factory=list)
    current_bowler: str = ""
    current_batters: list = field(default_factory=list)  # [striker, non-striker]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_BALL_RE = re.compile(r"<ball>(.*?)</ball>", re.DOTALL)
_KV_RE = re.compile(r"(\w+)=(\S+)")


def parse_ball_str(raw: str) -> Optional[Ball]:
    m = _BALL_RE.search(raw)
    if not m:
        return None
    kv = dict(_KV_RE.findall(m.group(1)))
    try:
        return Ball(
            over=float(kv.get("ov", 0)),
            bowler=kv.get("bwl", "unknown").replace("_", " "),
            batter=kv.get("bat", "unknown").replace("_", " "),
            non_striker=kv.get("nst", "unknown").replace("_", " "),
            runs=int(kv.get("runs", 0)),
            extras=kv.get("extras"),
            wicket=kv.get("wicket", "").replace("_", " ") or None,
            player_out=kv.get("out", "").replace("_", " ") or None,
            fielder=kv.get("fld", "").replace("_", " ") or None,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MatchEngine:
    MAX_OVERS = 20
    MAX_WICKETS = 10

    # Keep only the last N balls in context to avoid exceeding max_seq_len
    CONTEXT_WINDOW_BALLS = 60

    def __init__(self, checkpoint: str, device: str = "auto"):
        print(f"Loading model from {checkpoint}...")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device,
        )
        self.model.eval()
        self.context = ""
        self._match_header = ""
        self._ball_lines: list[str] = []   # rolling window of recent ball lines
        self._inn1_lines: list[str] = []   # full innings 1, pinned for inn2 context

    def start_match(
        self,
        team1: str,
        team2: str,
        venue: str,
        toss_winner: str,
        toss_decision: str,
        season: str = "2024",
        pitch: str = "flat",
    ) -> str:
        t1 = team1.replace(" ", "_")
        t2 = team2.replace(" ", "_")
        v = venue.replace(" ", "_")
        tw = toss_winner.replace(" ", "_")
        header = (
            f"<match> format=T20 venue={v} teams={t1},{t2} "
            f"toss={tw}:{toss_decision} season={season} pitch={pitch} </match>"
        )
        self._match_header = header + "\n"
        self._ball_lines = []
        self._inn1_lines = []
        self.context = self._match_header
        return header

    def start_innings(self, innings_num: int, batting_team: str) -> str:
        tag = f"inn{innings_num}"
        bt = batting_team.replace(" ", "_")
        line = f"<{tag}> batting={bt} </{tag}>"
        self._ball_lines.append(line)
        self._rebuild_context()
        return line

    def _rebuild_context(self):
        """Rebuild context: match header + pinned inn1 + rolling inn2 window.

        Total budget: CONTEXT_WINDOW_BALLS lines split between the two innings.
        Inn1 gets up to half the budget (last N balls of inn1 = death overs);
        inn2 rolling window fills the rest.
        """
        inn2_window = self.CONTEXT_WINDOW_BALLS // 2          # 30 balls for inn2
        inn1_window = self.CONTEXT_WINDOW_BALLS - inn2_window  # 30 balls for inn1

        parts = [self._match_header]
        if self._inn1_lines:
            parts.append("\n".join(self._inn1_lines[-inn1_window:]))
            parts.append("")   # blank line separator
        parts.append("\n".join(self._ball_lines[-inn2_window:]))
        self.context = "\n".join(parts) + "\n"

    def _compute_temperature(
        self,
        over: int,
        state,
        target: Optional[int],
        balls_since_wicket: int,
    ) -> float:
        temp = 1.0

        # --- Death overs (16-20): batters swing hard ---
        if over >= 15:
            base_boost = 0.1 * (over - 14)    # +0.1 at ov16, +0.5 at ov20
            # Extra aggression when wickets in hand (more to lose = swing harder)
            wickets_in_hand = self.MAX_WICKETS - state.wickets
            wicket_multiplier = 1.0 + (wickets_in_hand / self.MAX_WICKETS) * 0.5
            temp += base_boost * wicket_multiplier

        # --- Run pressure: slow scoring -> batters take risks ---
        recent_legal = [b for b in state.balls[-12:] if b and b.is_legal]
        if len(recent_legal) >= 6:
            recent_runs = sum(b.runs for b in recent_legal)
            if recent_runs <= 3:     # < 3 RPO last 2 overs — desperate
                temp += 0.25
            elif recent_runs <= 5:   # < 5 RPO — below par
                temp += 0.10

        # --- Chase pressure: required rate too high ---
        if target:
            needed = target - state.runs
            balls_remaining = (self.MAX_OVERS - over) * 6 - state.legal_balls
            if balls_remaining > 0:
                req_rpo = needed / balls_remaining * 6
                if req_rpo > 12:
                    temp += 0.25
                elif req_rpo > 9:
                    temp += 0.10

        # --- Wicket drought: boost if overdue (T20 avg ~17 balls/wicket) ---
        if balls_since_wicket >= 20:
            temp += 0.15 * ((balls_since_wicket - 20) // 6 + 1)

        return min(temp, 1.5)

    def generate_ball(
        self,
        over: int,
        ball_idx: int,
        bowler: str,
        batter: str,
        non_striker: str,
        temperature: float = 0.8,
        top_p: float = 0.9,
        score: Optional[int] = None,
        wickets: Optional[int] = None,
        target: Optional[int] = None,
    ) -> Optional[Ball]:
        # Build a hint prefix for the next ball to guide generation
        ov = f"{over}.{ball_idx + 1}"
        bwl = bowler.replace(" ", "_")
        bat = batter.replace(" ", "_")
        nst = non_striker.replace(" ", "_")
        prefix = f"<ball> ov={ov} bwl={bwl} bat={bat} nst={nst}"

        prompt = self.context + prefix
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                # Stop at newline — each ball is one line in training data
                eos_token_id=self.tokenizer.encode("\n", add_special_tokens=False)[-1],
            )

        generated = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        full_ball_str = prefix + " " + generated
        if "</ball>" not in full_ball_str:
            full_ball_str += " </ball>"

        ball = parse_ball_str(full_ball_str)
        line = full_ball_str.strip()
        self._ball_lines.append(line)
        self._rebuild_context()
        return ball

    def set_innings1_result(self, runs: int, wickets: int, overs: int) -> None:
        """Snapshot inn1 balls for context window; reset ball list for inn2.

        Note: do NOT inject a summary tag here — any tag format not present in
        training data (e.g. <inn1_result>) causes the model to output zeros in inn2.
        The last 30 balls of inn1 (pinned via _inn1_lines) already give the model
        full death-over context for the chase.
        """
        self._inn1_lines = list(self._ball_lines)
        self._ball_lines = []   # reset: inn2 rolls fresh
        self._rebuild_context()

    def add_result(self, winner: str, by: str) -> str:
        w = winner.replace(" ", "_")
        line = f"<result> winner={w} by={by} </result>"
        self._ball_lines.append(line)
        self._rebuild_context()
        return line

    def simulate_innings(
        self,
        innings_num: int,
        batting_team: str,
        bowling_team: str,
        target: Optional[int] = None,
        on_ball=None,
        on_over_end=None,
        on_wicket=None,
        get_bowler=None,
        get_next_batter=None,
        openers: Optional[list] = None,
    ) -> InningsState:
        """
        Simulate a full innings. Callbacks for interactive use:
            on_ball(ball, state)         — called after each delivery
            on_over_end(over, state)     — called after each complete over
            on_wicket(ball, state)       — called after a wicket
            get_bowler(over, state)      — return bowler name for the over
            get_next_batter(state)       — return next batter name
        """
        self.start_innings(innings_num, batting_team)
        state = InningsState(batting_team=batting_team, bowling_team=bowling_team)

        # Default squad placeholders if no callbacks
        if openers and len(openers) >= 2:
            state.current_batters = list(openers[:2])
        else:
            state.current_batters = ["Batter1", "Batter2"]
        state.current_bowler = "Bowler1"
        balls_since_wicket = 0  # track how long since last wicket

        for over in range(self.MAX_OVERS):
            if state.wickets >= self.MAX_WICKETS:
                break
            if target and state.runs >= target:
                break

            # Get bowler for this over
            if get_bowler:
                state.current_bowler = get_bowler(over, state)

            state.legal_balls = 0
            all_balls_this_over = 0  # includes wides/noballs

            while state.legal_balls < 6:
                if state.wickets >= self.MAX_WICKETS:
                    break

                batter = state.current_batters[0]
                non_striker = state.current_batters[1]

                temp = self._compute_temperature(
                    over, state, target, balls_since_wicket
                )

                ball = self.generate_ball(
                    over=over,
                    ball_idx=all_balls_this_over,
                    bowler=state.current_bowler,
                    batter=batter,
                    non_striker=non_striker,
                    temperature=temp,
                    score=state.runs,
                    wickets=state.wickets,
                    target=target,
                )

                all_balls_this_over += 1
                if ball is None:
                    state.legal_balls += 1
                    continue

                state.runs += ball.runs
                state.total_balls += 1
                if ball.is_legal:
                    state.legal_balls += 1
                state.balls.append(ball)

                if on_ball:
                    on_ball(ball, state)

                if ball.is_wicket:
                    state.wickets += 1
                    balls_since_wicket = 0
                    if on_wicket:
                        on_wicket(ball, state)
                    if get_next_batter and state.wickets < self.MAX_WICKETS:
                        state.current_batters[0] = get_next_batter(state)
                    else:
                        state.current_batters[0] = f"Batter{state.wickets + 2}"
                else:
                    if ball.is_legal:
                        balls_since_wicket += 1

                # Rotate strike on odd runs (wides don't rotate — penalty run, batter stays)
                if ball.runs % 2 == 1 and ball.extras != "wide":
                    state.current_batters.reverse()

                if target and state.runs >= target:
                    break

            # End of over: rotate strike
            state.current_batters.reverse()
            state.overs_complete += 1

            if on_over_end:
                on_over_end(over, state)

        return state


# ---------------------------------------------------------------------------
# CLI headless simulation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--team1", default="India")
    parser.add_argument("--team2", default="Pakistan")
    parser.add_argument("--venue", default="Wankhede")
    parser.add_argument("--toss-winner", default="India")
    parser.add_argument("--toss-decision", default="bat")
    parser.add_argument("--pitch", default="flat")
    args = parser.parse_args()

    engine = MatchEngine(args.checkpoint)
    engine.start_match(
        args.team1, args.team2, args.venue,
        args.toss_winner, args.toss_decision, pitch=args.pitch
    )

    def print_ball(ball, state):
        print(f"  {ball}")
        print(f"  Score: {state.runs}/{state.wickets} after {state.total_balls} balls")

    def print_over(over, state):
        print(f"\n--- End of over {over + 1} | {state.runs}/{state.wickets} ---")

    # Innings 1
    print(f"\n=== Innings 1: {args.team1} batting ===")
    inn1 = engine.simulate_innings(
        1, args.team1, args.team2,
        on_ball=print_ball, on_over_end=print_over
    )
    print(f"\n{args.team1} total: {inn1.runs}/{inn1.wickets} in {inn1.overs_complete} overs")

    # Innings 2
    target = inn1.runs + 1
    print(f"\n=== Innings 2: {args.team2} batting | Target: {target} ===")
    inn2 = engine.simulate_innings(
        2, args.team2, args.team1, target=target,
        on_ball=print_ball, on_over_end=print_over
    )
    print(f"\n{args.team2} total: {inn2.runs}/{inn2.wickets} in {inn2.overs_complete} overs")

    if inn2.runs >= target:
        winner = args.team2
        by = f"{10 - inn2.wickets} wickets"
    else:
        winner = args.team1
        by = f"{inn1.runs - inn2.runs} runs"

    result = engine.add_result(winner, by.replace(" ", ""))
    print(f"\n{result}")
    print(f"\nResult: {winner} won by {by}")


if __name__ == "__main__":
    main()
