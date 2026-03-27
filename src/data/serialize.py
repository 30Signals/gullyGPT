"""
Phase 1b: Convert T20 match JSONs into flat tagged string sequences for LM training.

Ball sequence format:
    <match> format=T20 venue=Wankhede teams=MI,CSK toss=MI:bat season=2023 </match>
    <inn1> batting=MI </inn1>
    <ball> ov=0.1 bwl=Bumrah bat=Rohit nst=Ishan runs=0 </ball>
    <ball> ov=0.3 bwl=Bumrah bat=Rohit nst=Ishan runs=0 wicket=caught out=Rohit fld=Dhoni </ball>
    <ball> ov=0.4 bwl=Bumrah bat=Kohli nst=Ishan extras=wide runs=1 </ball>
    <inn2> batting=CSK </inn2>
    ...
    <result> winner=MI by=23runs </result>

Usage:
    python src/data/serialize.py                    # full run
    python src/data/serialize.py --sample 5         # preview 5 sequences
    python src/data/serialize.py --index data/processed/t20_index.jsonl
"""

import argparse
import json
import random
from pathlib import Path

import tqdm


def serialize_match(data: dict) -> str:
    info = data.get("info", {})

    teams = info.get("teams", [])
    venue = info.get("venue", "unknown").replace(" ", "_")
    season = str(info.get("season", "")).replace("/", "-")
    match_type = info.get("match_type", "T20")
    toss = info.get("toss", {})
    toss_str = f"{toss.get('winner', '').replace(' ', '_')}:{toss.get('decision', '')}" if toss else "unknown"
    teams_str = ",".join(t.replace(" ", "_") for t in teams)

    lines = [
        f"<match> format={match_type} venue={venue} teams={teams_str} toss={toss_str} season={season} </match>"
    ]

    innings_list = data.get("innings", [])
    for inn_idx, innings in enumerate(innings_list):
        inn_tag = f"inn{inn_idx + 1}"
        batting_team = innings.get("team", "unknown").replace(" ", "_")
        lines.append(f"<{inn_tag}> batting={batting_team} </{inn_tag}>")

        for over_data in innings.get("overs", []):
            over_num = over_data.get("over", 0)
            for ball_idx, delivery in enumerate(over_data.get("deliveries", [])):
                ball_str = _serialize_delivery(over_num, ball_idx, delivery)
                lines.append(ball_str)

    # Result
    outcome = info.get("outcome", {})
    if "winner" in outcome:
        winner = outcome["winner"].replace(" ", "_")
        by = outcome.get("by", {})
        if "runs" in by:
            by_str = f"{by['runs']}runs"
        elif "wickets" in by:
            by_str = f"{by['wickets']}wickets"
        else:
            by_str = "unknown"
        lines.append(f"<result> winner={winner} by={by_str} </result>")
    elif "no_result" in outcome:
        lines.append("<result> no_result </result>")
    else:
        lines.append("<result> unknown </result>")

    return "\n".join(lines)


def _serialize_delivery(over_num: int, ball_idx: int, delivery: dict) -> str:
    batter = delivery.get("batter", "unknown").replace(" ", "_")
    bowler = delivery.get("bowler", "unknown").replace(" ", "_")
    non_striker = delivery.get("non_striker", "unknown").replace(" ", "_")
    runs = delivery.get("runs", {})
    total_runs = runs.get("total", 0)

    # over.ball notation (e.g. 12.3)
    ov = f"{over_num}.{ball_idx + 1}"

    parts = [f"ov={ov}", f"bwl={bowler}", f"bat={batter}", f"nst={non_striker}", f"runs={total_runs}"]

    # Extras
    extras = delivery.get("extras", {})
    for etype in ("wides", "noballs", "byes", "legbyes", "penalty"):
        if etype in extras:
            short = {"wides": "wide", "noballs": "noball", "byes": "bye",
                     "legbyes": "legbye", "penalty": "penalty"}[etype]
            parts.append(f"extras={short}")

    # Wickets
    wickets = delivery.get("wickets", [])
    for w in wickets:
        kind = w.get("kind", "unknown").replace(" ", "_")
        player_out = w.get("player_out", "").replace(" ", "_")
        fielders = w.get("fielders", [])
        fld_str = "_".join(f.get("name", "").replace(" ", "_") for f in fielders if isinstance(f, dict))
        parts.append(f"wicket={kind}")
        if player_out:
            parts.append(f"out={player_out}")
        if fld_str:
            parts.append(f"fld={fld_str}")

    return "<ball> " + " ".join(parts) + " </ball>"


def run(index_path: Path, out_dir: Path, sample: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(index_path) as f:
        records = [json.loads(line) for line in f]

    if sample:
        records = random.sample(records, min(sample, len(records)))
        for rec in records:
            with open(rec["path"]) as f:
                data = json.load(f)
            seq = serialize_match(data)
            print("\n" + "=" * 60)
            print(f"Match: {rec['teams']} | {rec['date']}")
            print("=" * 60)
            print(seq[:2000])
            print("...")
        return

    # Full run: write train.txt and val.txt (90/10 split by match)
    random.shuffle(records)
    split = int(len(records) * 0.9)
    splits = {"train": records[:split], "val": records[split:]}

    for split_name, split_records in splits.items():
        out_path = out_dir / f"{split_name}.txt"
        with open(out_path, "w") as out:
            for rec in tqdm.tqdm(split_records, desc=split_name):
                try:
                    with open(rec["path"]) as f:
                        data = json.load(f)
                    seq = serialize_match(data)
                    out.write(seq + "\n\n")
                except Exception as e:
                    print(f"Skipping {rec['path']}: {e}")

        print(f"{split_name}: {len(split_records)} matches -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/processed/t20_index.jsonl", type=Path)
    parser.add_argument("--out-dir", default="data/processed", type=Path)
    parser.add_argument("--sample", type=int, default=0,
                        help="Preview N random matches instead of full run")
    args = parser.parse_args()

    run(args.index, args.out_dir, args.sample)


if __name__ == "__main__":
    main()
