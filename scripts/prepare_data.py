"""Prepare private local JSONL inputs without changing the research scripts."""

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_cleaner():
    # `code` is a Python stdlib module, not a package in this repository.
    spec = importlib.util.spec_from_file_location(
        "twitter_cleaning", ROOT / "code" / "cleaning.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.clean_dataset


def deep_frame(raw, cleaned):
    """Retain only the fields main.py consumes; do not invent account identifiers."""
    import numpy as np
    import pandas as pd

    counts = ["user_statuses_count", "user_favourites_count", "user_listed_count"]
    columns = ["challenge_id", "text", *counts]
    if "label" in cleaned:
        columns.append("label")
    result = cleaned[columns].copy()
    users = raw["user"]
    if not users.map(lambda value: isinstance(value, dict)).all():
        raise ValueError("Deep preparation requires a user object in every record.")
    result["user_created_at"] = pd.to_datetime(
        users.map(lambda user: user.get("created_at")), utc=True, errors="coerce"
    )
    if result["user_created_at"].isna().any():
        raise ValueError("Deep preparation requires valid account creation timestamps.")
    if not np.isfinite(result[counts].to_numpy(dtype=float)).all():
        raise ValueError("Deep preparation requires finite account activity counts.")
    result["user_description"] = users.map(lambda user: user.get("description")).fillna("")
    result["user_location"] = users.map(lambda user: user.get("location"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/kaggle_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--deep", action="store_true", help="Also write *_multimodal.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="Replace derived output files")
    args = parser.parse_args()
    for path in (args.train, args.test):
        if not path.is_file():
            parser.error(f"Missing input: {path}. See docs/REPRODUCE.md for data access limits.")

    outputs = [args.output_dir / f"{split}_clean.jsonl" for split in ("train", "test")]
    if args.deep:
        outputs += [args.output_dir / f"{split}_multimodal.jsonl" for split in ("train", "test")]
    if set(path.resolve() for path in outputs) & {args.train.resolve(), args.test.resolve()}:
        parser.error("Output paths must not replace raw input files.")
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error("Output already exists; choose another --output-dir or use --overwrite.")

    import pandas as pd

    clean_dataset = load_cleaner()
    frames = []
    for path, is_train in ((args.train, True), (args.test, False)):
        raw = pd.read_json(path, lines=True, orient="records")
        if raw.empty or "challenge_id" not in raw or raw["challenge_id"].isna().any():
            parser.error("Inputs must contain records with nonmissing challenge_id values.")
        if is_train and ("label" not in raw or not raw["label"].isin([0, 1]).all()):
            parser.error("Training records must have binary numeric labels 0/1.")
        cleaned = clean_dataset(raw)
        if not is_train:
            cleaned = cleaned.drop(columns="label", errors="ignore")
        frames.append((raw, cleaned))

    prepared = [cleaned for _, cleaned in frames]
    if args.deep:
        prepared += [deep_frame(raw, cleaned) for raw, cleaned in frames]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, frame in zip(outputs, prepared):
        frame.to_json(path, orient="records", lines=True, date_format="iso")
        print(f"Saved {path} ({len(frame)} records)")


if __name__ == "__main__":
    main()
