## Twitter influence prediction

This repo contains two pipelines for the Twitter influence task:
- A tree-based baseline (`code/baseline.py`) on cleaned tabular/text features.
- A multimodal deep model (`code/main.py`) combining Transformer text with tabular metadata.

### Repository layout
- `code/cleaning.py` – feature engineering and cleaning helpers to turn raw tweet JSON into model-ready columns.
- `code/baseline.py` – TF-IDF + SVD + CatBoost / XGBoost ensemble, saves `output/submission.csv`.
- `code/main.py` – BERTweet + tabular attention model, saves `multimodal_influencer_model_final.pth` and `red.csv`.
- `data/` – place raw and cleaned JSONL files here (not tracked).
- `output/` – model outputs and submissions (created on run).
- `pyproject.toml` – dependency groups for uv.

## Setup (uv)

Install uv (pick one):
- Debian/Ubuntu (apt): `sudo apt update && sudo apt install -y uv` (if packaged) or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Arch (pacman): `sudo pacman -S uv` (if packaged) or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- macOS (Homebrew): `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows (PowerShell): `irm https://astral.sh/uv/install.ps1 | iex`
- Restart your shell, then check `uv --version`.

Dependency groups (`pyproject.toml`):
- Base: numpy, pandas, scikit-learn
- Random forests: catboost, xgboost
- Deep learning: torch, transformers
- Dev: matplotlib, ipykernel

Install what you need (examples):
- Base only: `uv sync`
- Base + random forests: `uv sync --group random-forests`
- Base + deep learning: `uv sync --group deep-learning`
- Base + dev tools: `uv sync --group dev`
- Everything: `uv sync --group random-forests --group deep-learning --group dev`
- Update lockfile after changes: `uv lock`

## Data preparation

Put the Kaggle files `train.jsonl` and `kaggle_test.jsonl` into the `data/` folder.

Baseline expects cleaned files named `data/train_clean.jsonl` and `data/test_clean.jsonl`.
The multimodal script defaults to Kaggle paths; change `TRAIN_PATH` / `TEST_PATH` in
`code/main.py` to point to your cleaned local files (e.g., `data/train_clean.jsonl`,
`data/test_clean.jsonl`).

To clean and rename the Kaggle files:
```bash
python - <<'PY'
import pandas as pd
from code.cleaning import clean_dataset

RAW_TRAIN = "data/train.jsonl"          # Kaggle train file
RAW_TEST = "data/kaggle_test.jsonl"     # Kaggle test file

train = pd.read_json(RAW_TRAIN, lines=True, orient="records")
test = pd.read_json(RAW_TEST, lines=True, orient="records")

train_clean = clean_dataset(train)
test_clean = clean_dataset(test)

train_clean.to_json("data/train_clean.jsonl", orient="records", lines=True)
test_clean.to_json("data/test_clean.jsonl", orient="records", lines=True)
print("Saved cleaned files to data/")
PY
```

## Baseline (CatBoost/XGBoost)
- Inputs: `data/train_clean.jsonl`, `data/test_clean.jsonl` (created above).
- Run: `python code/baseline.py`
- Output: `output/submission.csv` (ID, Predicted) using 5-fold CV and an ensemble of CatBoost + XGBoost with TF-IDF+SVD features and tabular columns.

## Multimodal deep model (BERTweet + tabular)
- Inputs: cleaned JSONL files; set `TRAIN_PATH`, `TEST_PATH`, `SUBMISSION_PATH` at the top of `code/main.py` to your local files (defaults point to Kaggle).
- Run: `python code/main.py`
- Hardware: GPU recommended (CUDA/MPS supported); falls back to CPU.
- Output: `multimodal_influencer_model_final.pth` (weights) and `red.csv` submission with predictions.

## Outputs
- Baseline: `output/submission.csv`
- Multimodal: `multimodal_influencer_model_final.pth`, `red.csv`

