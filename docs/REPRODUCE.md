# Reproduction guide

[README](../README.md) · [Methodology and evidence limits](METHODOLOGY.md)

The original research scripts and report are preserved. This guide distinguishes environment/preparation checks from a full experiment. **Raw data are absent; neither training pipeline nor any reported score has been reproduced in this audit.**

## Environment

Run commands from the repository root unless explicitly changing directories. Install [uv](https://docs.astral.sh/uv/getting-started/installation/). `.python-version` selects Python 3.13; `pyproject.toml` allows ≥3.13, but all allowed interpreter/platform combinations are not tested.

```bash
# Base preprocessing dependencies only.
uv sync --locked --no-default-groups

# Choose ONE additional environment as needed:
uv sync --locked --no-default-groups --group random-forests
uv sync --locked --no-default-groups --group deep-learning

# Or both model groups:
uv sync --locked --no-default-groups --group random-forests --group deep-learning
```

The historically named `random-forests` group contains **boosted trees** (CatBoost and XGBoost), not random forests. The `dev` group contains notebook/plotting tools and is optional. Synchronizing another selection can remove packages from the previous selection; pass the required groups again to `uv run`, or run the synchronized environment's Python directly.

The starting `uv.lock` treated all nine direct dependencies as mandatory despite the groups in `pyproject.toml`. Its metadata has been aligned with those groups, retaining all package versions. Use `--locked` to detect mismatches rather than `--frozen`, which skips the consistency check. Selected lock versions include NumPy 2.3.5, pandas 2.3.3, scikit-learn 1.8.0, CatBoost 1.2.8, XGBoost 3.1.2, Torch 2.9.1 and Transformers 4.57.3. These are dependency records, not proof of the historical experiment environment. Deep dependencies can be large on Linux because the lock includes CUDA libraries.

The deep script downloads `vinai/bertweet-base` tokenizer/config/weights through Transformers unless cached; it does not pin a checkpoint revision. Network access and storage are needed for that first load. The model card declares MIT licensing and describes English pretraining; consult the [model card](https://huggingface.co/vinai/bertweet-base) for provenance. Tokenizer/backend compatibility and any optional tokenizer packages need verification in the selected environment; no model download or tokenization was exercised in this audit.

## Obtain and arrange private inputs

The exact Kaggle competition/download URL, access rules and data license are **unverified**. The report calls this CSC-51054-EP's *Influencer or Observer* challenge; the notebook's `/kaggle/input/hollow-purple/` is only a local mount reference. Request the original files and terms from the course/challenge organizers. No guessed Kaggle slug or replacement corpus is supplied.

```text
data/
  train.jsonl          # labeled training records, numeric label 0/1
  kaggle_test.jsonl    # records for prediction; labels are not needed
```

Each file must contain one JSON object per line, using the original nested Twitter schema. `challenge_id` is required in both, and `label` is required in training. The cleaner directly accesses `user`, `quoted_status`, `text`, `created_at`, `entities`, `extended_entities`, `place`, `in_reply_to_screen_name`, `source`, and the four top-level engagement fields `retweet_count`, `favorite_count`, `reply_count`, `quote_count`. Missing **values** and missing **columns** are different: it can still require a field name even when challenge values are null. The `user` objects supply activity counts, creation timestamp, description and location; `extended_tweet` supplies full text/entities when present and truncated.

This is an implementation contract, not a verified raw-data schema. Do not manufacture missing challenge data or fill unavailable network information. Validate files locally against the actual organizer export. `.gitignore` excludes `data/` and JSONL files; it is not a substitute for checking staged files before publishing.

## Prepare baseline and deep inputs

The old README's `from code.cleaning import clean_dataset` fails in a normal interpreter: `code` resolves to Python's standard-library module, and this repository's `code/` has no `__init__.py`. The helper below loads the cleaner by absolute file path instead of importing that package name.

```bash
# Baseline inputs only:
uv run --locked --no-default-groups python scripts/prepare_data.py

# ALTERNATIVE for a fresh output directory: baseline + separate deep inputs.
uv run --locked --no-default-groups python scripts/prepare_data.py --deep
```

These are alternatives: the helper refuses to replace existing output files unless `--overwrite` is given. If you already prepared baseline files, either select another output directory or explicitly request replacement of the derived files. All checks and transformations finish before writing starts; interrupted disk writes are not atomic.

| Produced file | Consumer |
| --- | --- |
| `data/train_clean.jsonl` | Baseline training, including `label` |
| `data/test_clean.jsonl` | Baseline prediction, no `label` |
| `data/train_multimodal.jsonl` (with `--deep`) | Deep training; retained description/timestamp/location plus cleaned text and account counts |
| `data/test_multimodal.jsonl` (with `--deep`) | Deep prediction; same feature fields, no `label` |

The original cleaner returns only the baseline columns. Its output alone cannot feed `main.py`: it discards `user_created_at` and `user_description`, among other fields. The external adapter retrieves these from raw user objects without changing `code/cleaning.py`. It fills missing descriptions with empty strings and writes timestamps as ISO UTC so pandas recognizes `user_created_at` when loading JSONL. Deep preparation rejects missing/invalid account timestamps and nonfinite activity counts; it does not supply a reliable user identifier or fix the split heuristic.

Override raw paths explicitly, including on Kaggle:

```bash
uv run --locked --no-default-groups python scripts/prepare_data.py \
  --train /path/to/train.jsonl \
  --test /path/to/kaggle_test.jsonl \
  --output-dir data --deep

# Kaggle example: replace YOUR_AUTHORIZED_DATASET with the real mounted directory.
# The output directory must be writable; /kaggle/input is read-only.
uv run --locked --no-default-groups python scripts/prepare_data.py \
  --train /kaggle/input/YOUR_AUTHORIZED_DATASET/train.jsonl \
  --test /kaggle/input/YOUR_AUTHORIZED_DATASET/kaggle_test.jsonl \
  --output-dir /kaggle/working/prepared --deep
```

The mount name above is a placeholder, not a dataset identifier. On hosted notebooks, uv/interpreter setup and the repository checkout must be available before using these commands.

## Baseline execution: known blocker

The script uses literal paths relative to the **working directory**, not environment variables or CLI flags:

```text
data/train_clean.jsonl
data/test_clean.jsonl
output/submission.csv    # output columns: ID, Predicted
```

The following is the original entry point, **not a successful quickstart**:

```bash
uv run --locked --no-default-groups --group random-forests python code/baseline.py
```

Even with data/dependencies present, its `stop_words='french'` setting fails scikit-learn validation. Only `'english'`, a list or `None` is accepted ([documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)). The research source remains unchanged. Resolve that in a separately recorded experiment, using an explicit/versioned stop list or another declared setting; do not present the resulting run as an exact reproduction of the report. SVD's 200 dimensions and `min_df=5` also require adequate vocabulary size, so tiny sample data are not a training test.

To use a prepared directory elsewhere without editing this script, run its **absolute path** from a separate working directory containing `data/train_clean.jsonl` and `data/test_clean.jsonl` (copies or local symlinks). Outputs go to that working directory's `output/`. It will overwrite an existing submission; use a fresh working directory to preserve earlier outputs. `TRAIN_PATH`/`TEST_PATH` environment variables have no effect on this baseline.

## Deep execution: external path launcher

The unchanged `main.py` defaults are:

```text
TRAIN_PATH = /kaggle/input/hollow-purple/train_cleaned.jsonl
TEST_PATH  = /kaggle/input/hollow-purple/test_cleaned.jsonl
SUBMISSION_PATH = red.csv
```

Those `_cleaned` names differ from the baseline's `_clean` names; the deep model needs its own retained metadata schema. The external launcher imports the module and sets its path constants **after** import, then invokes `main()`. It changes no architecture, features, splitting, loss or training settings. Shell environment variables with the same names are not read by the original script.

After preparing data with `--deep` and installing the deep group:

```bash
uv run --locked --no-default-groups --group deep-learning python scripts/run_deep.py \
  --train data/train_multimodal.jsonl \
  --test data/test_multimodal.jsonl \
  --submission-path red.csv
```

For Kaggle, pass `/kaggle/working/prepared/train_multimodal.jsonl`, `/kaggle/working/prepared/test_multimodal.jsonl` and a writable submission path. These commands are conditional on real input files, compatible dependencies/tokenizer and adequate hardware; they have not been run through training here.

The original model always saves `multimodal_influencer_model_final.pth` in the **current working directory**, even when the submission path is overridden. `red.csv` contains `ID,Prediction` sorted by ID; the baseline uses `ID,Predicted`. Confirm the organizer's sample-submission schema before uploading either file. No official submission format is available here to resolve the difference.

The launcher refuses existing checkpoint/submission destinations to preserve previous runs. For repeat experiments, use a fresh working directory and absolute launcher/input paths, invoking the already synchronized repository environment's Python. It does not support resume, checkpoint-path overrides or automatic full-data refitting. CPU/MPS fallback exists in source but performance and backend compatibility are untested.

## Evaluation records to retain privately

Before treating a run as evidence, record the repository commit, input checksums, package versions, model revision, hardware, random seeds, split/group construction, feature configuration, thresholds and console logs. Preserve OOF/held-out predictions privately for auditing; do not commit tweets, descriptions or user identifiers. The current baseline does not save OOF arrays or fitted models, and the deep model saves weights without the full preprocessing/tokenizer configuration. See [methodology](METHODOLOGY.md) for threshold bias and split limitations.

## Verification scope for this documentation update

The documentation audit inspects the PDF and all three source files. Lightweight checks cover syntax, cleaner imports by file path, both helpers' help/missing-file behavior, and synthetic preparation/ISO-timestamp round trips. Synthetic records are private temporary test fixtures, not substitute challenge data, experiments or reported results. Local Markdown links and whitespace are checked; package versions in the updated lock are compared with the original lock.

The locked base environment was installed and checked with Python 3.13.11, NumPy 2.3.5, pandas 2.3.3 and scikit-learn 1.8.0. Its real `TfidfVectorizer.fit` rejected the baseline's exact TF-IDF configuration because of `stop_words='french'`. The deep `prepare_features` function was also exercised in isolation on synthetic prepared data without importing the training module. All 93 locked package versions match the starting lock, and the original scripts, PDF and license were checked byte for byte.

Full baseline fitting, deep module/model/tokenizer loading, GPU/backend execution, real-data preprocessing, official submissions and all reported validation scores remain **unexecuted/unverified**. The raw files, labels for evaluation, historical splits/logs and trained artifacts are not available.
