from __future__ import annotations

import torch

from deberta_multitask.model import DebertaV3MultiTaskScorer, MultiTaskConfig


def main() -> None:
    model = DebertaV3MultiTaskScorer(
        MultiTaskConfig(
            model_name="hf-internal-testing/tiny-random-deberta-v2",
            num_classes=5,
        )
    )
    batch = {
        "input_ids": torch.randint(0, 100, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
        "cls_labels": torch.tensor([1, 3]),
        "reg_labels": torch.tensor([0.25, 0.75]),
    }
    output = model(**batch)
    assert output.loss is not None
    assert output.logits.shape == (2, 5)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()

