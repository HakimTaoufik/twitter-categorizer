# Twitter/X influence classification

Predict whether the author of a **single tweet** is an **Influencer or Observer**, using tweet content and available account metadata. The [December 2025 challenge report](DL-data-challenge-HAK.pdf) defines these roles through asymmetric versus reciprocal follower networks; it says the author's explicit network counts and the tweet's engagement counts were removed. This is binary classification of an author's challenge-defined role, not a measurement of a tweet's future popularity.

The inputs are Twitter JSONL records from a Kaggle course challenge, with `label` in the training file and `challenge_id` identifying submission rows. Both scripts evaluate **accuracy** and assume numeric labels `0`/`1`; the available files do not establish which number means Influencer, the exact labeling rule, dataset size, or the official leaderboard metric. Raw training/test data and test labels are not included.

**Research question:** Does joint text + metadata modeling improve influence prediction versus strong tabular/TF-IDF baselines?

The repository implements two approaches to investigate this question. A controlled answer is still unverified: the scripts use different features and validation splits, and no run logs or evaluation predictions are tracked.

## Two approaches

```text
Baseline
Tweet + metadata → TF-IDF/SVD + encoding → XGBoost/CatBoost → Prediction

Deep model
Tweet + description + metadata → BERTweet + MLP → Attention fusion → Prediction
```

The baseline already combines text with metadata; it is not a metadata-only ablation. The deep model jointly trains a shared text encoder and an attention fusion network, rather than averaging independently trained classifiers. See [methodology](docs/METHODOLOGY.md) for exact features, hyperparameters and differences from the report's historical experiments.

## Results and evidence

**Current baseline OOF accuracy: not reported / cannot verify. Current attention-model validation accuracy: not reported / cannot verify.** No comparable results or ablation table can be established from the tracked artifacts.

The [report, Table 1, printed page 3](DL-data-challenge-HAK.pdf#page=4), reports historical validation accuracies of **0.839 for a user-only ensemble** and **0.840 for a late-fusion ensemble**. These are report transcriptions, not rerun results for the two scripts above. The report separately describes the attention experiment as overfitting, without a numerical result. Split membership, predictions and uncertainty estimates are unavailable; the reported difference does not establish an improvement or statistical significance.

Evaluation caveats:

- `baseline.py` uses **five stratified row folds**. Tweets from one author may appear in both training and validation. It tunes thresholds on each validation fold for printed fold scores, and on all OOF predictions for the model/ensemble scores and final submission. Scoring the threshold on the same OOF labels used to select it adds selection bias; a separate held-out evaluation is needed.
- `main.py` attempts an **80/20 user-level split**, grouping by SHA-256 of `user_created_at`. An account-creation timestamp is a proxy, not a unique user identifier: distinct accounts can share it, missing values collapse together, and inconsistent representations can separate one account. Hashing does not guarantee leakage prevention. Its scores are not directly comparable to the baseline's row-wise OOF scores.

## Data, access and licensing

Expected local inputs are **`data/train.jsonl`** (labeled training records) and **`data/kaggle_test.jsonl`** (prediction records). They are absent from this checkout. The report and old setup instructions identify Kaggle as the source, but an exact challenge/download URL and data license could not be verified. The `/kaggle/input/hollow-purple/` paths in `main.py` are notebook mount paths, not a verified public dataset source. Obtain authorized files and rules from the original course/challenge organizers; no substitute download is supplied here.

Keep raw and cleaned tweet/user records, predictions and weights local. The [repository license](LICENSE) is GNU AGPL v3; it does not grant rights to redistribute challenge data or third-party model weights. The [BERTweet model card](https://huggingface.co/vinai/bertweet-base) identifies an English-tweet pretrained checkpoint and declares MIT licensing. Dataset language composition is unverified; the report also discusses French text models, so language suitability needs checking.

## Quickstart

Run from the repository root. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. Python **3.13** is selected by `.python-version` (`pyproject.toml` allows Python ≥3.13).

```bash
uv sync --locked --no-default-groups
uv run --locked --no-default-groups python scripts/prepare_data.py --help
```

After placing authorized raw files at the expected paths:

```bash
# Writes baseline inputs and separate inputs that retain the deep model's fields.
# Omit --deep to prepare only baseline inputs.
uv run --locked --no-default-groups python scripts/prepare_data.py --deep
```

The helper loads `code/cleaning.py` by file path, avoiding the stdlib `code` module conflict. It refuses existing output files unless `--overwrite` is explicitly supplied. Custom raw paths are accepted with `--train` and `--test`; `--output-dir` changes the destination.

**Training is conditional, not verified end to end.** The unchanged baseline has an invalid `stop_words='french'` setting. The original cleaner alone drops the deep model's required fields; `--deep` prepares separate compatible inputs, and [the path launcher](scripts/run_deep.py) overrides its Kaggle paths without editing the script. See [reproduction instructions](docs/REPRODUCE.md) for commands, dependency setup, output filenames, known blockers and what was actually tested.

## Limits and next experiments

A fair comparison needs the same audited author groups, held-out rows, feature availability and tuning budget for all approaches. Reevaluate text-only, metadata-only and combined models on that split; select thresholds only within the training portion. Audit quoted-user network/engagement features against challenge rules, measure group overlap and class balance, and report repeated-seed variability and group-level confidence intervals. The current evidence does not isolate the benefit of attention or descriptions. See [methodology](docs/METHODOLOGY.md) for the proposed controls.

## Repository guide

| File | Purpose |
| --- | --- |
| [code/cleaning.py](code/cleaning.py) | Original JSONL cleaning and feature engineering |
| [code/baseline.py](code/baseline.py) | Original TF-IDF/SVD + XGBoost/CatBoost OOF pipeline |
| [code/main.py](code/main.py) | Original BERTweet + metadata attention model and training |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Task, architecture details, evidence and evaluation limits |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Data preparation, execution paths and verification scope |
| [DL-data-challenge-HAK.pdf](DL-data-challenge-HAK.pdf) | Preserved team report, December 11, 2025 |

Report authors: Al Hakim Taoufik, Aya Bayna and Khalid Akhmouch (team HAK, CSC-51054-EP).
