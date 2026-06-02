from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED_COLUMNS = {
    "response_id",
    "essay_set",
    "question",
    "reference_answer",
    "student_answer",
    "score",
}

EXCEL_ERROR_VALUES = {
    "#DIV/0!",
    "#N/A",
    "#NAME?",
    "#NULL!",
    "#NUM!",
    "#REF!",
    "#VALUE!",
}


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        rows = list(reader)

    cleaned = []
    for row in rows:
        question = row["question"].strip()
        reference_answer = row["reference_answer"].strip()
        student_answer = row["student_answer"].strip()
        if not question or not reference_answer or not student_answer:
            continue
        if student_answer.upper() in EXCEL_ERROR_VALUES:
            continue

        essay_set = int(row["essay_set"])
        score = int(float(row["score"]))
        cleaned.append(
            {
                "response_id": row["response_id"],
                "essay_set": essay_set,
                "question": question,
                "reference_answer": reference_answer,
                "student_answer": student_answer,
                "score": score,
                "score2_inter_rater": int(float(row.get("score2_inter_rater") or score)),
            }
        )
    return cleaned


def add_labels(rows: list[dict]) -> list[dict]:
    max_score_by_set = defaultdict(int)
    for row in rows:
        max_score_by_set[row["essay_set"]] = max(max_score_by_set[row["essay_set"]], row["score"])

    labeled = []
    for row in rows:
        max_score = max_score_by_set[row["essay_set"]]
        if max_score <= 0:
            raise ValueError(f"essay_set {row['essay_set']} has max score 0")
        item = dict(row)
        item["cls_label"] = row["score"]
        item["reg_label"] = row["score"] / max_score
        item["max_score"] = max_score
        item["stratify_label"] = f"essay_set_{row['essay_set']}_score_{row['score']}"
        labeled.append(item)
    return labeled


def split_grouped(
    rows: list[dict],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    groups_by_stratum = defaultdict(dict)

    for row in rows:
        key = normalize_answer(row["student_answer"])
        groups_by_stratum[row["stratify_label"]].setdefault(key, []).append(row)

    train = []
    validation = []
    for groups in groups_by_stratum.values():
        group_items = list(groups.values())
        rng.shuffle(group_items)
        target_validation_size = round(sum(len(group) for group in group_items) * validation_ratio)

        current_validation_size = 0
        prompt_train = []
        prompt_validation = []
        for group in group_items:
            if current_validation_size < target_validation_size:
                prompt_validation.extend(group)
                current_validation_size += len(group)
            else:
                prompt_train.extend(group)

        train.extend(prompt_train)
        validation.extend(prompt_validation)

    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def write_jsonl(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keep_columns = [
        "response_id",
        "essay_set",
        "question",
        "reference_answer",
        "student_answer",
        "score",
        "score2_inter_rater",
        "max_score",
        "cls_label",
        "reg_label",
        "stratify_label",
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            output = {column: row[column] for column in keep_columns}
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")


def summarize(rows: list[dict], name: str) -> None:
    print(f"\n{name}: {len(rows):,} rows")
    by_prompt = Counter(row["essay_set"] for row in rows)
    by_score = Counter(row["cls_label"] for row in rows)
    print("Rows by essay_set:", dict(sorted(by_prompt.items())))
    print("Rows by score:", dict(sorted(by_score.items())))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ASAP-SAS CSV for multi-task training.")
    parser.add_argument("--input", required=True, help="Path to asap_sas_reference_augmented_full.csv")
    parser.add_argument("--output-dir", default="data", help="Directory for train.jsonl and validation.jsonl")
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.input)
    output_dir = Path(args.output_dir)

    rows = add_labels(read_rows(csv_path))
    train, validation = split_grouped(rows, args.validation_ratio, args.seed)

    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(validation, output_dir / "validation.jsonl")

    max_cls_label = max(row["cls_label"] for row in rows)
    metadata = {
        "source_csv": str(csv_path),
        "rows": len(rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "num_classes": max_cls_label + 1,
        "score_max_by_essay_set": dict(
            sorted({row["essay_set"]: row["max_score"] for row in rows}.items())
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summarize(rows, "Full dataset")
    summarize(train, "Train")
    summarize(validation, "Validation")
    print(f"\nWrote {output_dir / 'train.jsonl'}")
    print(f"Wrote {output_dir / 'validation.jsonl'}")
    print(f"Wrote {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
