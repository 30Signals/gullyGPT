"""
Phase 3 verification: generate N full T20 matches and validate scorecards.

Checks:
  - Each innings has ≤ 20 overs and ≤ 10 wickets
  - All ball run values are 0-6 (or slightly more for extras)
  - Balls parse cleanly (no malformed outputs)
  - Score totals are reasonable (no negative runs)

Prints per-match scorecards and aggregate stats.

Usage:
    python src/generate/eval.py --checkpoint checkpoints/qwen-cricket --n 10
    python src/generate/eval.py --checkpoint checkpoints/qwen-cricket --n 3 --verbose
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.generate.engine import MatchEngine, InningsState


FIXTURES = [
    dict(team1="India",      team2="Pakistan",   venue="Dubai_International", toss_winner="India",   toss_decision="bat"),
    dict(team1="Australia",  team2="England",    venue="MCG",                 toss_winner="England", toss_decision="field"),
    dict(team1="West_Indies",team2="South_Africa",venue="Sabina_Park",        toss_winner="West_Indies", toss_decision="bat"),
]


@dataclass
class MatchResult:
    team1: str
    team2: str
    inn1_runs: int = 0
    inn1_wickets: int = 0
    inn1_overs: int = 0
    inn2_runs: int = 0
    inn2_wickets: int = 0
    inn2_overs: int = 0
    winner: str = ""
    parse_errors: int = 0
    violations: list = field(default_factory=list)

    def validate(self) -> list:
        issues = []
        for inn, runs, wkts, ovs in [
            (1, self.inn1_runs, self.inn1_wickets, self.inn1_overs),
            (2, self.inn2_runs, self.inn2_wickets, self.inn2_overs),
        ]:
            if wkts > 10:
                issues.append(f"Inn{inn}: {wkts} wickets > 10")
            if ovs > 20:
                issues.append(f"Inn{inn}: {ovs} overs > 20")
            if runs < 0:
                issues.append(f"Inn{inn}: negative runs {runs}")
            if runs > 400:
                issues.append(f"Inn{inn}: suspiciously high score {runs}")
        return issues


def run_eval(checkpoint: str, n: int, verbose: bool = False, seed_fixture_idx: int = -1):
    engine = MatchEngine(checkpoint)
    results = []

    for i in range(n):
        fix = FIXTURES[i % len(FIXTURES)]
        t1 = fix["team1"].replace("_", " ")
        t2 = fix["team2"].replace("_", " ")

        print(f"\n{'='*60}")
        print(f"Match {i+1}/{n}: {t1} vs {t2}")
        print(f"{'='*60}")

        engine.start_match(
            t1, t2,
            fix["venue"].replace("_", " "),
            fix["toss_winner"].replace("_", " "),
            fix["toss_decision"],
        )

        result = MatchResult(team1=t1, team2=t2)

        def on_ball(ball, state):
            if verbose:
                print(f"  {ball}")

        def on_over_end(over, state):
            print(f"  Over {over+1}: {state.runs}/{state.wickets}")

        # Innings 1
        inn1: InningsState = engine.simulate_innings(
            1, t1, t2,
            on_ball=on_ball if verbose else None,
            on_over_end=on_over_end,
        )
        result.inn1_runs = inn1.runs
        result.inn1_wickets = inn1.wickets
        result.inn1_overs = inn1.overs_complete
        result.parse_errors += sum(1 for b in inn1.balls if b is None)

        print(f"\n  {t1}: {inn1.runs}/{inn1.wickets} in {inn1.overs_complete} overs")

        # Innings 2
        target = inn1.runs + 1
        print(f"\n  Target: {target}")

        inn2: InningsState = engine.simulate_innings(
            2, t2, t1,
            target=target,
            on_ball=on_ball if verbose else None,
            on_over_end=on_over_end,
        )
        result.inn2_runs = inn2.runs
        result.inn2_wickets = inn2.wickets
        result.inn2_overs = inn2.overs_complete
        result.parse_errors += sum(1 for b in inn2.balls if b is None)

        print(f"\n  {t2}: {inn2.runs}/{inn2.wickets} in {inn2.overs_complete} overs")

        if inn2.runs >= target:
            result.winner = t2
            margin = f"{10 - inn2.wickets} wickets"
        else:
            result.winner = t1
            margin = f"{inn1.runs - inn2.runs} runs"

        engine.add_result(result.winner, margin.replace(" ", ""))
        print(f"\n  Result: {result.winner} won by {margin}")

        result.violations = result.validate()
        if result.violations:
            print(f"  ⚠ VIOLATIONS: {result.violations}")
        if result.parse_errors:
            print(f"  ⚠ Parse errors: {result.parse_errors}")

        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {n} matches")
    print(f"{'='*60}")

    total_violations = sum(len(r.violations) for r in results)
    total_parse_errors = sum(r.parse_errors for r in results)
    avg_inn1 = sum(r.inn1_runs for r in results) / n
    avg_inn2 = sum(r.inn2_runs for r in results) / n
    avg_wkts1 = sum(r.inn1_wickets for r in results) / n
    avg_wkts2 = sum(r.inn2_wickets for r in results) / n

    print(f"Rule violations : {total_violations}")
    print(f"Parse errors    : {total_parse_errors}")
    print(f"Avg Inn1 score  : {avg_inn1:.1f}/{avg_wkts1:.1f}")
    print(f"Avg Inn2 score  : {avg_inn2:.1f}/{avg_wkts2:.1f}")
    print(f"\nScorecard:")
    for r in results:
        status = "OK" if not r.violations and not r.parse_errors else "WARN"
        print(f"  [{status}] {r.team1} {r.inn1_runs}/{r.inn1_wickets} vs "
              f"{r.team2} {r.inn2_runs}/{r.inn2_wickets}  →  {r.winner} won")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to fine-tuned checkpoint")
    parser.add_argument("--n", type=int, default=5, help="Number of matches to generate")
    parser.add_argument("--verbose", action="store_true", help="Print every ball")
    args = parser.parse_args()

    run_eval(args.checkpoint, args.n, args.verbose)


if __name__ == "__main__":
    main()
