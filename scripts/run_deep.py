"""Override input/submission paths for the unchanged multimodal training script."""

import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train_multimodal.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/test_multimodal.jsonl"))
    parser.add_argument("--submission-path", type=Path, default=Path("red.csv"))
    args = parser.parse_args()
    for path in (args.train, args.test):
        if not path.is_file():
            parser.error(f"Missing input: {path}. Prepare data with scripts/prepare_data.py --deep.")
    checkpoint = Path("multimodal_influencer_model_final.pth")
    if args.submission_path.exists() or checkpoint.exists():
        parser.error("An output already exists; use a fresh working directory to preserve it.")
    if args.submission_path.resolve() == checkpoint.resolve():
        parser.error("Submission and checkpoint paths must be different.")

    source = Path(__file__).resolve().parents[1] / "code" / "main.py"
    spec = importlib.util.spec_from_file_location("twitter_multimodal", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.TRAIN_PATH = str(args.train.resolve())
    module.TEST_PATH = str(args.test.resolve())
    module.SUBMISSION_PATH = str(args.submission_path.resolve())
    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    module.main()


if __name__ == "__main__":
    main()
