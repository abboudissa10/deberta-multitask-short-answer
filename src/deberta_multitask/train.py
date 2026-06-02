from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

from deberta_multitask.data import DataConfig, load_jsonl_dataset, tokenize_dataset
from deberta_multitask.model import DebertaV3MultiTaskScorer, MultiTaskConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    training_config = config["training"]
    data_config = config["data"]

    set_seed(training_config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        use_fast=config.get("tokenizer_use_fast", False),
    )
    dataset = load_jsonl_dataset(
        DataConfig(
            train_file=data_config["train_file"],
            validation_file=data_config["validation_file"],
            max_length=training_config["max_length"],
        )
    )
    tokenized = tokenize_dataset(dataset, tokenizer, training_config["max_length"])

    model = DebertaV3MultiTaskScorer(
        MultiTaskConfig(
            model_name=config["model_name"],
            num_classes=data_config["num_classes"],
        )
    )

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        warmup_ratio=training_config["warmup_ratio"],
        logging_steps=training_config["logging_steps"],
        evaluation_strategy=training_config["eval_strategy"],
        save_strategy=training_config["save_strategy"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    trainer.save_model(config["output_dir"])


if __name__ == "__main__":
    main()
