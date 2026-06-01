"""Convert private-set model outputs (JSONL) into the submission CSV (id,response)."""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "results" / "initial_private_results.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "csv" / "submission.csv"


def convert(input_path: Path, output_path: Path) -> int:
    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj or "response" not in obj:
                raise KeyError(f"line {lineno}: missing 'id' or 'response' field")
            rows.append((int(obj["id"]), obj["response"]))

    rows.sort(key=lambda r: r[0])

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        writer.writerows(rows)

    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    n = convert(args.input, args.output)
    print(f"wrote {n} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
