from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

pd.set_option("display.max_columns", 200)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
print("Using device:", DEVICE)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = "/kaggle/input/hollow-purple/"
#OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = "/kaggle/input/hollow-purple/train_cleaned.jsonl"
TEST_PATH = "/kaggle/input/hollow-purple/test_cleaned.jsonl"
SUBMISSION_PATH = "red.csv"


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def clean_tweet(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+", "URL", text)
    text = re.sub(r"@\w+", "USER", text)
    return text.strip()


def make_user_id(created_at_val) -> str:
    """Stable hash of created_at to identify a user."""
    return hashlib.sha256(str(created_at_val).encode("utf-8")).hexdigest()


def prepare_features(df: pd.DataFrame, max_created: pd.Timestamp) -> pd.DataFrame:
    """Compute tabular features and ensure description/location helpers."""
    df = df.copy()
    df["account_age_days_user"] = (max_created - df["user_created_at"]).dt.days.clip(
        lower=1
    )
    months = df["account_age_days_user"] / 30.0
    months = months.replace(0, 1 / 30)
    df["statuses_per_month"] = df["user_statuses_count"] / months
    df["favourites_per_month"] = df["user_favourites_count"] / months
    df["user_description_len"] = df["user_description"].fillna("").astype(str).str.len()
    if "user_location" in df.columns:
        df["user_has_location"] = df["user_location"].notna().astype(int)
    else:
        df["user_has_location"] = 0
    return df


# ---------------------------------------------------------------------------
# Multimodal attention model
# ---------------------------------------------------------------------------
class MultiModalInfluencerModel(nn.Module):
    """
    Attention-based multimodal model that combines:
      - Tweet text (encoded by a Transformer)
      - User description text (encoded by the same Transformer)
      - Tabular metadata features (dense vector)
    Fuses with Multihead self-attention over 3 tokens.
    """

    def __init__(
        self,
        text_model_name: str = "vinai/bertweet-base",
        num_tabular_features: int = 6,
        fusion_hidden_dim: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()

        self.text_config = AutoConfig.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_hidden_size = self.text_config.hidden_size

        self.tweet_proj = nn.Linear(self.text_hidden_size, fusion_hidden_dim)
        self.desc_proj = nn.Linear(self.text_hidden_size, fusion_hidden_dim)

        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, fusion_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_hidden_dim),
        )

        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=fusion_hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.fusion_ffn = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 2 * fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * fusion_hidden_dim, fusion_hidden_dim),
        )
        self.fusion_ln1 = nn.LayerNorm(fusion_hidden_dim)
        self.fusion_ln2 = nn.LayerNorm(fusion_hidden_dim)

        self.cls_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb

    def forward(
        self,
        tweet_input_ids,
        tweet_attention_mask,
        desc_input_ids,
        desc_attention_mask,
        tabular_feats,
        labels=None,
    ):
        tweet_cls = self.encode_text(tweet_input_ids, tweet_attention_mask)
        desc_cls = self.encode_text(desc_input_ids, desc_attention_mask)

        tweet_tok = self.tweet_proj(tweet_cls)
        desc_tok = self.desc_proj(desc_cls)
        tab_tok = self.tabular_mlp(tabular_feats)

        tokens = torch.stack([tweet_tok, desc_tok, tab_tok], dim=1)
        attn_out, _ = self.fusion_attn(tokens, tokens, tokens)
        tokens = self.fusion_ln1(tokens + attn_out)

        ffn_out = self.fusion_ffn(tokens)
        tokens = self.fusion_ln2(tokens + ffn_out)

        fused = tokens[:, 0, :]
        logits = self.cls_head(fused).squeeze(-1)

        loss = None
        if labels is not None:
            labels = labels.float()
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        return {"logits": logits, "loss": loss}


class MultiModalDataset(Dataset):
    """
    Dataset wrapper for the multimodal model.
    Expects columns: tweet_text, user_description, tabular_cols, label.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        tabular_cols: Iterable[str],
        text_col: str = "tweet_text",
        desc_col: str = "user_description",
        max_len: int = 128,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.tabular_cols = list(tabular_cols)
        self.text_col = text_col
        self.desc_col = desc_col
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        tweet_text = str(row.get(self.text_col, ""))
        desc_text = str(row.get(self.desc_col, ""))

        tweet_enc = self.tokenizer(
            tweet_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        desc_enc = self.tokenizer(
            desc_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        tab_feats = torch.tensor(
            row[self.tabular_cols].astype(float).values, dtype=torch.float
        )
        label_val = row.get("label", None)
        label = (
            torch.tensor(int(label_val), dtype=torch.float)
            if label_val is not None and not pd.isna(label_val)
            else None
        )

        sample = {
            "tweet_input_ids": tweet_enc["input_ids"].squeeze(0),
            "tweet_attention_mask": tweet_enc["attention_mask"].squeeze(0),
            "desc_input_ids": desc_enc["input_ids"].squeeze(0),
            "desc_attention_mask": desc_enc["attention_mask"].squeeze(0),
            "tabular_feats": tab_feats,
        }
        if label is not None:
            sample["labels"] = label
        return sample


def create_multimodal_loaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    tokenizer,
    tabular_cols: Iterable[str],
    max_len: int = 128,
    train_batch_size: int = 16,
    val_batch_size: int = 16,
):
    train_ds = MultiModalDataset(
        df_train,
        tokenizer=tokenizer,
        tabular_cols=tabular_cols,
        max_len=max_len,
    )
    val_ds = MultiModalDataset(
        df_val,
        tokenizer=tokenizer,
        tabular_cols=tabular_cols,
        max_len=max_len,
    )
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate_multimodal(model, data_loader: DataLoader) -> tuple[float, float]:
    model.eval()
    ncorrect = 0
    ntotal = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            labels = batch.get("labels")
            for k, v in batch.items():
                if k != "labels" and v is not None:
                    batch[k] = v.to(DEVICE)
            if labels is not None:
                labels = labels.to(DEVICE)
                batch["labels"] = labels
            outputs = model(**batch)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            if labels is not None:
                total_loss += outputs["loss"].item() * labels.size(0)
                ntotal += labels.size(0)
                ncorrect += (preds.view(-1) == labels.long().view(-1)).sum()
    acc = (ncorrect.float() / ntotal).item() if ntotal > 0 else 0.0
    avg_loss = (total_loss / ntotal) if ntotal > 0 else 0.0
    return acc, avg_loss


def train_multimodal_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    lr: float = 2e-5,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        # for running training accuracy
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)

            labels = batch.get("labels")

            # move tensors to device
            for k, v in batch.items():
                if k != "labels" and v is not None and hasattr(v, "to"):
                    batch[k] = v.to(DEVICE)

            if labels is not None:
                labels = labels.to(DEVICE)
                batch["labels"] = labels

            outputs = model(**batch)
            loss = outputs["loss"]

            # backward + optimization step with amp
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if labels is not None:
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # compute running training accuracy (assuming classification logits)
                logits = outputs.get("logits", None)
                if logits is not None:
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).long()
                    correct = (preds.view(-1) == labels.long().view(-1)).sum().item()
                    train_correct += correct
                    train_total += batch_size

            # verbose logging every 100 batches (and at the end)
            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                avg_loss_so_far = (
                    total_loss / total_samples if total_samples > 0 else 0.0
                )
                train_acc_so_far = (
                    train_correct / train_total if train_total > 0 else 0.0
                )
                print(
                    f"[Epoch {epoch}] Batch {batch_idx}/{len(train_loader)} "
                    f"batch_loss={loss.item():.4f} "
                    f"avg_loss={avg_loss_so_far:.4f} "
                    f"train_acc={train_acc_so_far:.4f}"
                )

        train_loss = total_loss / total_samples if total_samples > 0 else 0.0
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        val_acc, val_loss = evaluate_multimodal(model, val_loader)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    final_acc, _ = evaluate_multimodal(model, val_loader)
    print(f"Final validation accuracy: {final_acc:.4f}")


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
def main():
    print("=== Starting multimodal influencer pipeline ===")
    # Load data
    train_data = pd.read_json(TRAIN_PATH, lines=True, orient="records")
    test_data = pd.read_json(TEST_PATH, lines=True, orient="records")
    print(f"Loaded train rows: {len(train_data):,} | test rows: {len(test_data):,}")

    # Use the extended text column directly
    train_data = train_data.rename(columns={"text": "tweet_text"})
    test_data = test_data.rename(columns={"text": "tweet_text"})

    # Hash user_id from created_at to prevent cross-user leakage
    train_data["user_id"] = train_data["user_created_at"].apply(make_user_id)
    test_data["user_id"] = test_data["user_created_at"].apply(make_user_id)
    print("Generated user_id from created_at for train/test.")

    # Feature engineering
    max_created = train_data["user_created_at"].max()
    train_data = prepare_features(train_data, max_created)
    test_data = prepare_features(test_data, max_created)
    print("Computed tabular features.")

    # Clean tweet text
    train_data["tweet_text"] = train_data["tweet_text"].apply(clean_tweet)
    test_data["tweet_text"] = test_data["tweet_text"].apply(clean_tweet)
    print("Cleaned tweet text.")

    # User-level split to avoid leakage
    user_labels = (
        train_data.groupby("user_id")["label"].mean().round().astype(int).reset_index()
    )
    train_users, val_users = train_test_split(
        user_labels["user_id"],
        test_size=0.2,
        random_state=42,
        stratify=user_labels["label"],
    )
    print(
        f"User-level split -> train users: {len(train_users):,}, "
        f"val users: {len(val_users):,}"
    )
    train_df = train_data[train_data["user_id"].isin(train_users)].reset_index(
        drop=True
    )
    val_df = train_data[train_data["user_id"].isin(val_users)].reset_index(drop=True)
    print(
        f"Train tweets: {len(train_df):,} | Val tweets: {len(val_df):,} | "
        f"Class balance (train) -> pos={train_df['label'].mean():.3f}"
    )

    # Tokenizer and model
    text_model_name = "vinai/bertweet-base"
    tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)
    print(f"Loaded tokenizer: {text_model_name}")

    tabular_cols = [
        "account_age_days_user",
        "statuses_per_month",
        "favourites_per_month",
        "user_listed_count",
        "user_description_len",
        "user_has_location",
    ]

    train_loader, val_loader = create_multimodal_loaders(
        train_df,
        val_df,
        tokenizer=tokenizer,
        tabular_cols=tabular_cols,
        max_len=128,
        train_batch_size=16,
        val_batch_size=16,
    )
    print(
        f"Dataloaders ready -> train batches: {len(train_loader):,}, "
        f"val batches: {len(val_loader):,}"
    )

    model = MultiModalInfluencerModel(
        text_model_name=text_model_name,
        num_tabular_features=len(tabular_cols),
        fusion_hidden_dim=256,
        num_heads=4,
    ).to(DEVICE)
    print("Model instantiated and moved to device.")

    # Train with per-epoch accuracy
    print("=== Training start ===")
    train_multimodal_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        lr=2e-5,
    )
    print("=== Training complete ===")

    # ---- save trained model ----
    model_path = "multimodal_influencer_model_final.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")
    # ---------------------------------

    # Test inference
    print("Preparing test dataloader...")
    test_ds = MultiModalDataset(
        test_data,
        tokenizer=tokenizer,
        tabular_cols=tabular_cols,
        text_col="tweet_text",
        desc_col="user_description",
        max_len=128,
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    print(f"Test batches: {len(test_loader):,}")

    model.eval()
    print("=== Inference on test ===")
    all_preds: List[int] = []
    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                if k != "labels" and v is not None:
                    batch[k] = v.to(DEVICE)
            outputs = model(**batch)
            probs = torch.sigmoid(outputs["logits"])
            preds = (probs >= 0.5).long().cpu().numpy()
            all_preds.append(preds)
    y_test_pred = np.concatenate(all_preds, axis=0)

    submission = (
        test_data.assign(Prediction=y_test_pred.astype(int))
        [["challenge_id", "Prediction"]]
        .rename(columns={"challenge_id": "ID"})
        .sort_values("ID")
        .reset_index(drop=True)
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to {SUBMISSION_PATH}")
    print(submission.head())
    print("=== Done ===")


if __name__ == "__main__":
    main()