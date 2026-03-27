"""
Phase 1a: Filter T20/IT20 matches from the full Cricsheet dataset.
Outputs data/processed/t20_index.jsonl — one JSON line per match with path + metadata.

Usage:
    python src/data/filter.py
    python src/data/filter.py --data-dir data/all_json --out data/processed/t20_index.jsonl
"""

import argparse
import json
import os
from pathlib import Path

import tqdm


KEEP_TYPES = {"T20", "IT20"}


def build_index(data_dir: Path, out_path: Path) -> int:
    files = sorted(data_dir.glob("*.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out_path, "w") as out:
        for fpath in tqdm.tqdm(files, desc="Filtering"):
            if fpath.name == "README.txt":
                continue
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except Exception:
                continue

            info = data.get("info", {})
            match_type = info.get("match_type", "")
            if match_type not in KEEP_TYPES:
                continue

            teams = info.get("teams", [])
            dates = info.get("dates", [])
            record = {
                "path": str(fpath),
                "match_id": fpath.stem,
                "match_type": match_type,
                "gender": info.get("gender", ""),
                "season": info.get("season", ""),
                "venue": info.get("venue", ""),
                "city": info.get("city", ""),
                "teams": teams,
                "date": dates[0] if dates else "",
                "event": info.get("event", {}).get("name", ""),
                "outcome": info.get("outcome", {}),
            }
            out.write(json.dumps(record) + "\n")
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/all_json", type=Path)
    parser.add_argument("--out", default="data/processed/t20_index.jsonl", type=Path)
    args = parser.parse_args()

    count = build_index(args.data_dir, args.out)
    print(f"\nDone. {count} T20/IT20 matches indexed -> {args.out}")


if __name__ == "__main__":
    main()
