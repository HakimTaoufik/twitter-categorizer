from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.base import clone

import numpy as np
import pandas as pd

import os

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load cleaned train set
train_data = pd.read_json('data/train_clean.jsonl', lines=True, orient='records')

# Define feature groups
numeric_features = [
    'user_statuses_count',
    'user_favourites_count',
    'user_listed_count',
    'account_age_days',
    'user_description_len',
    'text_length_chars',
    'text_length_words',
    'n_hashtags',
    'n_mentions',
    'n_urls',
    'n_media',
    'n_photo',
    'n_video',
    'n_gif',
    'tweet_hour',
    'tweet_dow',
    'quoted_retweet_count',
    'quoted_favorite_count',
    'quoted_reply_count',
    'quoted_quote_count',
    'quoted_user_followers_count',
    'quoted_user_friends_count',
    'hashtags_per_word',
    'mentions_per_word',
    'urls_per_word',
    'media_per_word',
]

bool_features = [
    'is_reply',
    'is_quote',
    'is_standalone',
    'has_media',
    'has_hashtag',
    'has_mention',
    'has_url',
    'has_place',
    'user_has_location',
    'is_weekend',
    'quoted_user_verified',
    'source_is_twitter',
]

categorical_features = [
    'source',
    'source_label',
    'source_domain',
    'source_family',
    'device_class',
    'device_os',
    'device_detail',
    'browser_family',
    'place_country',
    'place_country_code',
    'place_type',
    'quoted_lang',
]

text_feature = 'text'

# Text vectorizer + dimensionality reduction
text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        max_features=80000,
        sublinear_tf=True,
        stop_words='french',
    )),
    ('svd', TruncatedSVD(
        n_components=200,
        random_state=42
    ))
])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

bool_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, text_feature),
        ('num',  numeric_transformer, numeric_features),
        ('bool', bool_transformer, bool_features),
        ('cat',  categorical_transformer, categorical_features),
    ],
    remainder='drop'
)

# Split features/labels
X = train_data.copy()
y = train_data['label'].values
X = X.drop(columns=['label'])

# Base model configs
pos_ratio = y.mean()
scale_pos_weight = (1 - pos_ratio) / pos_ratio

xgb_clf = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=400,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    random_state=42,
)

cat_clf = CatBoostClassifier(
    iterations=400,
    learning_rate=0.1,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
    task_type="CPU",
)

pipeline_xgb = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf', xgb_clf)
])

pipeline_cat = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf', cat_clf)
])


# Search best accuracy threshold on probas
def find_best_threshold(y_true, probs, step=0.01):
    thresholds = np.arange(0.0, 1.0 + step, step)
    best_thr, best_acc = 0.5, 0.0
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_thr, best_acc


# Stratified K-fold training/inference helper
def run_kfold(model, model_name, X_df, y_arr, X_test_df, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_arr))
    test_fold_preds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_df, y_arr), 1):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]

        model_fold = clone(model)
        model_fold.fit(X_tr, y_tr)

        oof_preds[va_idx] = model_fold.predict_proba(X_va)[:, 1]
        test_fold_preds.append(model_fold.predict_proba(X_test_df)[:, 1])

        fold_thr, fold_acc = find_best_threshold(y_va, oof_preds[va_idx])
        print(f"[{model_name}] Fold {fold}: acc@best_thr={fold_acc:.4f} thr={fold_thr:.3f}")

    best_thr, best_acc = find_best_threshold(y_arr, oof_preds)
    print(f"[{model_name}] OOF acc@best_thr={best_acc:.4f} thr={best_thr:.3f}")

    test_mean = np.mean(test_fold_preds, axis=0)
    return oof_preds, test_mean, best_thr


# Load cleaned test set
X_test = pd.read_json("data/test_clean.jsonl", lines=True, orient='records')

# Run CV for each model
oof_xgb, test_xgb, _ = run_kfold(pipeline_xgb, "XGBoost", X, y, X_test, n_splits=5)
oof_cat, test_cat, _ = run_kfold(pipeline_cat, "CatBoost", X, y, X_test, n_splits=5)

# Ensemble averaged probas
ensemble_oof = (oof_xgb + oof_cat) / 2.0
ens_thr, ens_acc = find_best_threshold(y, ensemble_oof)
print(f"[Ensemble] OOF acc@best_thr={ens_acc:.4f} thr={ens_thr:.3f}")

ensemble_test = (test_xgb + test_cat) / 2.0
final_preds = (ensemble_test >= ens_thr).astype(int)

# Write submission
os.makedirs("output", exist_ok=True)
submission = pd.DataFrame({
    "ID": X_test["challenge_id"],
    "Predicted": final_preds
})

submission.to_csv("output/submission.csv", index=False)
print("Saved submission.csv")