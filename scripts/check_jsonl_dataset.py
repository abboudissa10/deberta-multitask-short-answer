from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REQUIRED_COLUMNS = {
    "question",
    "reference_answer",
    "student_answer",
    "cls_label",
    "reg_label",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_COLUMNS - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing columns {sorted(missing)}")
            rows.append(row)
    return rows


def summarize(path: Path) -> None:
    rows = read_jsonl(path)
    labels = Counter(int(row["cls_label"]) for row in rows)
    prompts = Counter(int(row.get("essay_set", -1)) for row in rows)
    reg_values = [float(row["reg_label"]) for row in rows]

    print(f"\n{path}: {len(rows):,} rows")
    print("Class labels:", dict(sorted(labels.items())))
    print("Essay sets:", dict(sorted(prompts.items())))
    print(f"reg_label min={min(reg_values):.4f} max={max(reg_values):.4f}")
    if min(reg_values) < 0 or max(reg_values) > 1:
        raise ValueError(f"{path} has reg_label values outside [0, 1]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check prepared JSONL files.")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    for path in args.paths:
        summarize(Path(path))


if __name__ == "__main__":
    main()
