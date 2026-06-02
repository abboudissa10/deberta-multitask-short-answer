from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


@dataclass
class MultiTaskConfig:
    model_name: str
    num_classes: int
    dropout: float = 0.15


class DebertaV3MultiTaskScorer(nn.Module):
    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(config.dropout)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.num_classes),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, 1),
        )

        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cls_labels: torch.Tensor | None = None,
        reg_labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])

        logits = self.classification_head(pooled)
        regression = self.regression_head(pooled).squeeze(-1)

        loss = None
        if cls_labels is not None and reg_labels is not None:
            cls_loss = nn.CrossEntropyLoss()(logits, cls_labels.long())
            reg_loss = nn.SmoothL1Loss()(regression, reg_labels.float())
            loss = (
                torch.exp(-self.log_var_cls) * cls_loss
                + self.log_var_cls
                + torch.exp(-self.log_var_reg) * reg_loss
                + self.log_var_reg
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    @torch.no_grad()
    def predict_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        alpha: float = 0.6,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classification_head(pooled)
        regression = self.regression_head(pooled).squeeze(-1).sigmoid()

        probs = logits.softmax(dim=-1)
        class_values = torch.linspace(0, 1, logits.size(-1), device=logits.device)
        expected_class_score = (probs * class_values).sum(dim=-1)

        return alpha * regression + (1 - alpha) * expected_class_score

