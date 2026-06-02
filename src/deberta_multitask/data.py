from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class DataConfig:
    train_file: str
    validation_file: str
    max_length: int


def load_jsonl_dataset(config: DataConfig) -> DatasetDict:
    data_files = {
        "train": str(Path(config.train_file)),
        "validation": str(Path(config.validation_file)),
    }
    return load_dataset("json", data_files=data_files)


def build_text(example: dict) -> str:
    question = example.get("question") or example.get("prompt") or ""
    reference = example.get("reference_answer") or example.get("rubric") or ""
    answer = example.get("student_answer") or example.get("answer") or example.get("essay_text") or ""

    parts = []
    if question:
        parts.append(f"Question: {question}")
    if reference:
        parts.append(f"Reference answer: {reference}")
    parts.append(f"Student answer: {answer}")
    return "\n".join(parts)


def tokenize_dataset(dataset: DatasetDict, tokenizer: PreTrainedTokenizerBase, max_length: int) -> DatasetDict:
    def tokenize(example: dict) -> dict:
        encoded = tokenizer(
            build_text(example),
            truncation=True,
            max_length=max_length,
        )
        encoded["cls_labels"] = int(example["cls_label"])
        encoded["reg_labels"] = float(example["reg_label"])
        return encoded

    return dataset.map(tokenize, remove_columns=dataset["train"].column_names)

