# DeBERTa-v3 Multi-Task Short Answer Scoring

This project trains a DeBERTa-v3 model with two heads:

- classification head for discrete answer labels or score buckets
- regression head for normalized continuous scores

The target datasets are ASAP-SAS and SciEntsBank. The goal is to compare this multi-task model against DeBERTa-v3 single-head classification and single-head regression baselines.

## Project Layout

```text
.
├── configs/
│   └── default.yaml
├── scripts/
│   └── smoke_test.py
├── src/
│   └── deberta_multitask/
│       ├── data.py
│       ├── metrics.py
│       ├── model.py
│       └── train.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
└── requirements.txt
```

## Quick Start

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run a smoke test:

```bash
python scripts/smoke_test.py
```

## Prepare ASAP-SAS

Upload your prepared CSV into the repository or Codespaces workspace. If the file is named:

```text
asap_sas_reference_augmented_full.csv
```

run:

```bash
python scripts/prepare_asap_sas.py \
  --input asap_sas_reference_augmented_full.csv \
  --output-dir data \
  --validation-ratio 0.15 \
  --seed 42
```

This creates:

```text
data/train.jsonl
data/validation.jsonl
data/metadata.json
```

The script makes the labels expected by the model:

- `cls_label`: the original integer ASAP score
- `reg_label`: the score normalized per `essay_set`

For example, a score of `1` becomes:

- `0.3333` when the prompt maximum is `3`
- `0.5` when the prompt maximum is `2`

Check the prepared files:

```bash
python scripts/check_jsonl_dataset.py data/train.jsonl data/validation.jsonl
```

Train:

```bash
python -m deberta_multitask.train --config configs/default.yaml
```

## GitHub Setup

From this project folder:

```bash
git init
git add .
git commit -m "Initial DeBERTa multi-task short answer scorer"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/deberta-multitask-short-answer.git
git push -u origin main
```

Then open the repo on GitHub and check the `Actions` tab. The CI workflow runs the smoke test.

## Dataset Notes

The training code expects JSONL rows with at least:

```json
{"question":"...","reference_answer":"...","student_answer":"...","cls_label":2,"reg_label":0.6667}
```

Recommended input format:

```text
Question: ...
Reference answer: ...
Student answer: ...
```

For ASAP-SAS:

- classification label: integer score bucket
- regression label: score normalized to `[0, 1]` within each `essay_set`

For SciEntsBank:

- classification label: original class label
- regression label: ordinal mapped label, normalized to `[0, 1]`

## Baselines To Beat

Train and compare:

1. DeBERTa-v3 with regression head only
2. DeBERTa-v3 with classification head only
3. DeBERTa-v3 with both heads

Use the same train/validation/test split and same random seeds.
