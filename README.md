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

You will likely need to adapt `src/deberta_multitask/data.py` to the exact local or Hugging Face dataset fields you use.

Recommended input format:

```text
Question: ...
Reference answer: ...
Student answer: ...
```

For ASAP-SAS:

- regression label: score normalized to `[0, 1]`
- classification label: integer score bucket

For SciEntsBank:

- classification label: original class label
- regression label: ordinal mapped label, normalized to `[0, 1]`

## Baselines To Beat

Train and compare:

1. DeBERTa-v3 with regression head only
2. DeBERTa-v3 with classification head only
3. DeBERTa-v3 with both heads

Use the same train/validation/test split and same random seeds.

