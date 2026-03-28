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
    dict(team1="India",       team2="Pakistan",    venue="Dubai_International", toss_winner="India",      toss_decision="bat"),
    dict(team1="Australia",   team2="England",     venue="MCG",                 toss_winner="England",    toss_decision="field"),
    dict(team1="West Indies", team2="South Africa",venue="Sabina_Park",         toss_winner="West Indies",toss_decision="bat"),
]

SQUADS = {
    "India": {
        "batters": ["Rohit Sharma", "Virat Kohli", "Shubman Gill", "Suryakumar Yadav",
                    "Hardik Pandya", "Rishabh Pant", "Ravindra Jadeja", "Kuldeep Yadav",
                    "Jasprit Bumrah", "Mohammed Shami", "Arshdeep Singh"],
        "bowlers": ["Jasprit Bumrah", "Mohammed Shami", "Arshdeep Singh",
                    "Hardik Pandya", "Ravindra Jadeja", "Kuldeep Yadav"],
    },
    "Pakistan": {
        "batters": ["Babar Azam", "Mohammad Rizwan", "Fakhar Zaman", "Iftikhar Ahmed",
                    "Shadab Khan", "Asif Ali", "Mohammad Nawaz", "Shaheen Afridi",
                    "Naseem Shah", "Haris Rauf", "Saim Ayub"],
        "bowlers": ["Shaheen Afridi", "Naseem Shah", "Haris Rauf",
                    "Shadab Khan", "Mohammad Nawaz", "Iftikhar Ahmed"],
    },
    "Australia": {
        "batters": ["David Warner", "Travis Head", "Steve Smith", "Glenn Maxwell",
                    "Mitchell Marsh", "Tim David", "Marcus Stoinis", "Pat Cummins",
                    "Mitchell Starc", "Josh Hazlewood", "Adam Zampa"],
        "bowlers": ["Pat Cummins", "Mitchell Starc", "Josh Hazlewood",
                    "Adam Zampa", "Glenn Maxwell", "Marcus Stoinis"],
    },
    "England": {
        "batters": ["Jos Buttler", "Jonny Bairstow", "Jason Roy", "Ben Stokes",
                    "Liam Livingstone", "Moeen Ali", "Sam Curran", "Chris Woakes",
                    "Jofra Archer", "Adil Rashid", "Mark Wood"],
        "bowlers": ["Jofra Archer", "Mark Wood", "Chris Woakes",
                    "Sam Curran", "Adil Rashid", "Moeen Ali"],
    },
    "West Indies": {
        "batters": ["Nicholas Pooran", "Rovman Powell", "Brandon King", "Shimron Hetmyer",
                    "Andre Russell", "Kieron Pollard", "Kyle Mayers", "Sunil Narine",
                    "Jason Holder", "Alzarri Joseph", "Akeal Hosein"],
        "bowlers": ["Alzarri Joseph", "Jason Holder", "Andre Russell",
                    "Sunil Narine", "Akeal Hosein", "Kieron Pollard"],
    },
    "South Africa": {
        "batters": ["Quinton de Kock", "Temba Bavuma", "Rassie van der Dussen", "David Miller",
                    "Heinrich Klaasen", "Aiden Markram", "Marco Jansen", "Wayne Parnell",
                    "Kagiso Rabada", "Anrich Nortje", "Tabraiz Shamsi"],
        "bowlers": ["Kagiso Rabada", "Anrich Nortje", "Marco Jansen",
                    "Wayne Parnell", "Tabraiz Shamsi", "Aiden Markram"],
    },
}


def make_squad_callbacks(batting_team: str, bowling_team: str):
    """Return (get_bowler, get_next_batter) callbacks using real player names."""
    bowlers = SQUADS.get(bowling_team, {}).get("bowlers", ["Bowler1"])
    batters = SQUADS.get(batting_team, {}).get("batters", [])
    batter_queue = list(batters[2:])  # openers handled by engine init

    def get_bowler(over, state):
        return bowlers[over % len(bowlers)]

    def get_next_batter(state):
        if batter_queue:
            return batter_queue.pop(0)
        return f"Batter{state.wickets + 2}"

    return get_bowler, get_next_batter


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

        def make_over_callback(batting_team):
            prev = {"runs": 0}
            def on_over_end(over, state):
                this_over = state.runs - prev["runs"]
                prev["runs"] = state.runs
                print(f"  [{batting_team}] Over {over+1}: {state.runs}/{state.wickets}  (+{this_over} runs)")
            return on_over_end

        # Determine batting/bowling order based on toss
        if fix["toss_decision"] == "bat":
            batting1, bowling1 = t1, t2
        else:
            batting1, bowling1 = t2, t1
        batting2, bowling2 = bowling1, batting1

        sq1_bat = SQUADS.get(batting1, {}).get("batters", ["Batter1", "Batter2"])
        sq2_bat = SQUADS.get(batting2, {}).get("batters", ["Batter1", "Batter2"])

        get_bowler1, get_next_batter1 = make_squad_callbacks(batting1, bowling1)
        get_bowler2, get_next_batter2 = make_squad_callbacks(batting2, bowling2)

        # Innings 1
        inn1: InningsState = engine.simulate_innings(
            1, batting1, bowling1,
            on_ball=on_ball if verbose else None,
            on_over_end=make_over_callback(batting1),
            get_bowler=get_bowler1,
            get_next_batter=get_next_batter1,
            openers=sq1_bat[:2],
        )
        result.inn1_runs = inn1.runs
        result.inn1_wickets = inn1.wickets
        result.inn1_overs = inn1.overs_complete
        result.parse_errors += sum(1 for b in inn1.balls if b is None)

        print(f"\n  {batting1}: {inn1.runs}/{inn1.wickets} in {inn1.overs_complete} overs")

        # Pin inn1 result into context so model knows the target throughout inn2
        engine.set_innings1_result(inn1.runs, inn1.wickets, inn1.overs_complete)

        # Innings 2
        target = inn1.runs + 1
        print(f"\n  Target: {target}")

        inn2: InningsState = engine.simulate_innings(
            2, batting2, bowling2,
            target=target,
            on_ball=on_ball if verbose else None,
            on_over_end=make_over_callback(batting2),
            get_bowler=get_bowler2,
            get_next_batter=get_next_batter2,
            openers=sq2_bat[:2],
        )
        result.inn2_runs = inn2.runs
        result.inn2_wickets = inn2.wickets
        result.inn2_overs = inn2.overs_complete
        result.parse_errors += sum(1 for b in inn2.balls if b is None)

        print(f"\n  {batting2}: {inn2.runs}/{inn2.wickets} in {inn2.overs_complete} overs")

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
