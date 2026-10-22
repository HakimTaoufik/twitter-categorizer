# Methodology and evidence audit

[README](../README.md) · [Reproduction guide](REPRODUCE.md) · [Original report](../DL-data-challenge-HAK.pdf)

## Task and available evidence

The report, *Influencer or Observer: Predicting Social Roles* (team HAK, December 11, 2025), describes binary prediction of an author's role from one tweet. It characterizes Influencers by highly asymmetric follower networks and Observers by more reciprocal networks. The exact numerical labeling rule and mapping of `0`/`1` to these role names are not documented in the available artifacts.

The unit of training, validation accuracy and submission is a **tweet record**, not an aggregated account. The report mentions three to five tweets per user; this cannot be checked without the data. The raw files are described as anonymized Kaggle challenge exports. Neither an exact challenge URL nor train/test sizes, class balance, language distribution, collection period or redistribution license can be verified from this checkout. There are no raw records, run logs, OOF predictions, split manifests, submissions or trained weights in the tracked tree at the audited starting commit `12b76ce`.

Training scripts expect `label` values in `{0, 1}` and use `challenge_id` for submission IDs. Both compute accuracy (correct predictions / evaluated tweet records). The baseline's XGBoost objective uses log loss, CatBoost specifies an AUC evaluation metric, and the deep model trains with binary cross-entropy; those settings do not establish the official challenge metric. Test labels and official leaderboard results are unavailable.

**Question:** Does joint text + metadata modeling improve influence prediction versus strong tabular/TF-IDF baselines? The repository contains implementations relevant to this question, but its present artifacts do not support a controlled comparison or an attention ablation.

## Data pipeline

[cleaning.py](../code/cleaning.py) copies the input DataFrame and returns a new feature frame. Its docstring incorrectly says that it mutates the input and returns nothing; the executable behavior returns `dfc`.

1. Drop redundant identifiers/timestamps and several sparse raw fields. Convert tweet/account timestamps, counts, flags and text types.
2. Extract account activity, description, location presence and creation time from nested `user` objects. The author's follower/friend counts and identity extraction are commented out.
3. Extract quoted-tweet language, engagement and quoted-author network counts/verification. These are different from the main author's removed network features.
4. Prefer `extended_tweet.full_text` and its entities when `truncated` is true. Count hashtags, mentions, URLs and media types; derive reply/quote/standalone flags, word/character lengths and per-word densities.
5. Compute UTC tweet hour/day, weekend flag and account age at tweet time. Parse source HTML into client/domain, coarse source family, device/OS and broad browser categories. These are heuristics, not measured device/browser identities.
6. Compute additional fields such as emoji/punctuation counts, log activity, capitalization share and quoted-author follower ratio. Drop constants except the required baseline columns; normalize the baseline feature types.
7. **Return only** identifiers/label/text plus the baseline contract: 26 numeric, 12 boolean and 12 categorical features. Many intermediate fields, including description text, account creation timestamp and location text, do not survive this final selection.

Cleaning is applied separately to train and test. The preserved baseline contract protects required columns from constant removal; other fields are not generally aligned by this procedure. Some raw columns are indexed directly even when mostly missing, so arbitrary Twitter exports are not guaranteed to work.

[prepare_data.py](../scripts/prepare_data.py) is an external adapter. It calls the unchanged cleaner via its file path. With `--deep`, it also writes separate frames containing the cleaned full tweet text, the three account counts, IDs/labels and `user_created_at`, `user_description`, `user_location` recovered from the original nested user objects. It writes ISO UTC account timestamps, fills missing descriptions with empty text, and rejects missing/invalid timestamps or nonfinite account counts for deep inputs. It does not create an identity key, impute deep numeric features, normalize metadata or alter either model.

## A. TF-IDF/SVD plus boosted trees

[baseline.py](../code/baseline.py) combines tweet text and metadata in a scikit-learn `ColumnTransformer`. It is already a joint-feature baseline, not a text-free control.

| Feature block | Fields / processing |
| --- | --- |
| Tweet text | Lowercase word unigrams/bigrams; TF-IDF with `min_df=5`, `max_df=0.9`, `max_features=80000`, sublinear TF; TruncatedSVD with 200 components, seed 42 |
| Account numeric | `user_statuses_count`, `user_favourites_count`, `user_listed_count`, `account_age_days`, `user_description_len` |
| Tweet numeric | Character/word lengths; hashtag/mention/URL counts; media/photo/video/GIF counts; UTC hour/day; four per-word densities |
| Quote numeric | Retweet/favorite/reply/quote counts and quoted-author follower/friend counts |
| Boolean (12) | Reply, quote, standalone, media, hashtag, mention, URL, place, user location, weekend, quoted-author verified, Twitter source |
| Categorical (12) | Source HTML, source label/domain/family, device class/OS/detail, browser family, place country/code/type, quoted language |

Numeric features use median imputation and standard scaling; booleans use most-frequent imputation; categorical features use most-frequent imputation and one-hot encoding with unknown categories ignored. CatBoost receives this encoded matrix rather than native categorical columns. Unselected columns, including labels/IDs, are dropped by the transformer.

**Execution blocker:** the code passes `stop_words='french'` to `TfidfVectorizer`. [scikit-learn 1.8 documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) allows only `'english'`, a list or `None`; a fit with `'french'` raises a parameter-validation error. No French stop list is supplied. The research scripts are preserved, so this setting remains unchanged. Choosing a real list or disabling stop-word removal would define a new, explicitly recorded experiment.

| Model | Checked-in settings |
| --- | --- |
| XGBoost | 400 trees, learning rate 0.1, depth 6, subsample 0.8, column subsample 0.8, histogram trees, binary logistic objective, log-loss metric, all CPU threads, seed 42 |
| CatBoost | 400 iterations, learning rate 0.1, depth 8, Logloss objective, AUC metric, CPU, seed 42, no file logging |

XGBoost's positive-class weight is `(1 - y.mean()) / y.mean()`, calculated once from all training labels, rather than inside each fold. Both classes must be present. There is no Optuna search, validation-based early stopping or training-time evaluation set in this script, despite these methods appearing in the historical report.

Each model is cloned and fitted separately within five `StratifiedKFold` row splits (`shuffle=True`, seed 42). Thus learned TF-IDF, SVD, imputers, scaler and category encodings are fitted within each training fold. Each row gets an out-of-fold (OOF) probability from a model not trained on that row; each test row gets the mean of the five fold predictions for each classifier. The final ensemble averages XGBoost and CatBoost probabilities with equal weights. It does not retrain one final model on all labeled rows.

Thresholds from 0 to 1 in steps of 0.01 maximize accuracy. Each printed fold accuracy is optimized on that fold's validation labels; model-level and ensemble OOF accuracies are optimized on the full OOF labels. The ensemble threshold is then applied to the averaged test probabilities. Ties retain the first (lowest) threshold. These scores include threshold selection on the scoring labels; OOF prediction generation alone does not remove that bias.

## B. Shared BERTweet with metadata attention fusion

[main.py](../code/main.py) uses `vinai/bertweet-base` for both tweet text and user descriptions. This is the [English-tweet checkpoint](https://huggingface.co/vinai/bertweet-base), not the French model also mentioned in the report. Input URLs and mentions in tweets are replaced with `URL` and `USER`; descriptions are passed through as text. The tokenizer pads/truncates **each** sequence to 128 tokens.

The same trainable Transformer is called twice. Its first-token hidden vector is taken for each modality, then projected separately to 256 dimensions. The tabular encoder is `Linear(6, 256) → ReLU → LayerNorm`, using:

- `account_age_days_user`: days between the latest account creation time in the full labeled input and each account creation time, clipped below at 1. This is **not** account age at tweet time.
- `statuses_per_month`, `favourites_per_month`: activity counts divided by this age / 30.
- `user_listed_count`, description character length and location presence (`notna()`, so an empty string still counts as present).

There is no numeric standardization or general imputation in this model. The maximum creation timestamp is computed before the validation split, so even this unlabeled preprocessing reference uses validation records. A strict held-out design would estimate it on training records alone.

The two projected text vectors and encoded tabular vector form **three tokens** of size 256. Four-head self-attention mixes the tokens, followed by residual addition/LayerNorm, then a `256 → 512 → 256` ReLU feed-forward block with another residual/LayerNorm. The fused **tweet token** alone feeds `Linear(256, 256) → ReLU → Dropout(0.1) → Linear(256, 1)`.

Training uses `BCEWithLogitsLoss`, AdamW at `2e-5`, batch size 16, three epochs and gradient-norm clipping at 1.0. The encoder is trainable throughout; there is no freeze/unfreeze schedule or early stopping. CUDA `GradScaler` is used, but no autocast context is present; this does not implement a complete mixed-precision training path. Device selection tries CUDA, then MPS, then CPU; full execution on these backends has not been verified here. Torch/NumPy/DataLoader random seeds are not set, so the split seed alone does not make training deterministic.

Validation and test prediction use sigmoid probability ≥0.5. Accuracy is weighted by tweet count even though the split is grouped. The final epoch's weights are saved, not the best validation checkpoint; there is no full-training-data refit. Only a state dictionary is saved, without tokenizer, preprocessing reference, split or model revision metadata.

## Split heuristic and comparability

The deep script computes `SHA256(str(user_created_at))` as `user_id`. It groups rows on that value and obtains a group label by rounding the mean of the group's labels. It stratifies those group labels into an 80/20 train/validation split with seed 42, then selects all rows assigned to each side.

This is an **attempted user-level split**, not validated author separation. Distinct accounts created at the same recorded instant share a key; all equally represented missing timestamps share a key; inconsistent timestamps/representations for the same account can yield different keys. A hash preserves these ambiguities. Rounded group labels can hide conflicting labels in collided groups. The external preparation helper rejects invalid timestamps, but does not establish identity uniqueness or eliminate collisions.

The baseline's stratified row folds can place the same author's tweets on both sides. The deep model's proxy groups have different sample composition and training fraction, and its six metadata features differ substantially from the baseline's 50 features. Therefore even recovered scores for these scripts would not isolate architecture quality or fusion gains. A repeated-author overlap audit needs authorized stable identifiers or organizer-provided groups; deriving or publishing real identities is unnecessary.

## What the report actually measures

The report's Table 1 (printed page 3, PDF page 4) labels its numbers **validation accuracy**. It lists TF-IDF + logistic regression at 0.625, a fine-tuned Transformer at 0.836, user-only XGBoost at 0.833, user-only CatBoost at 0.837, a user-only ensemble at 0.839, early fusion at 0.838 and late fusion at 0.840. Section 2.4.1 separately mentions approximately 80% accuracy for TabNet. These are historical report statements, not independently verified measurements.

The report says its two text-only models used the same row-wise split. It does not supply row indices or sufficient configuration/prediction artifacts to recreate a common evaluation across all listed methods. Its late-fusion method averages independent model probabilities, whereas `main.py` performs representation-level attention fusion. The attention experiment is discussed separately as overfitting with no numerical score. The report also mentions autoencoder embeddings, tuning and transformer freezing that are not implemented in the checked-in attention script.

Consequently, **current script scores: not reported / cannot verify**. No controlled results/ablation table is presented. There is no supported SOTA, significance, improved-accuracy or successful-reproduction claim. The original PDF is retained unchanged as historical evidence, including its stronger claims; this audit does not adopt them.

## Next experiments needed to answer the question

1. Obtain the official role definition, numeric label mapping, metric, data rules and authorized files. Audit quoted-author follower/friend counts, quoted engagement and account listed counts against the intended feature restrictions; their actual availability cannot be established from code alone.
2. Create one audited author-group split and an untouched evaluation set for every model. Report group/row counts, class balance, duplicates, timestamp missingness, proxy collisions and overlap. Consider a temporal holdout if deployment concerns future accounts/tweets.
3. Fit all learned preprocessing and reference timestamps inside training folds. Select thresholds and ensemble weights only on inner validation data, then evaluate once on the untouched outer holdout. Report both fixed-0.5 and training-selected thresholds.
4. Under the same rows and tuning budget, compare metadata-only trees, TF-IDF-only trees, combined TF-IDF/metadata trees, tweet-only BERTweet, tweet + description BERTweet, and text + metadata with concatenation versus attention. Match feature availability where testing architecture rather than feature choice.
5. Save configurations, package/model revisions, split manifests and private OOF probabilities. Report accuracy and class-wise errors, repeated-seed variability, group-level confidence intervals, runtime and hardware. Keep raw records and identifiers private. An ablation table is appropriate only after these comparable evaluations exist.
