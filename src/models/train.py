"""
train.py
--------
Main training entrypoint:
1. Loads noisy customer support tickets dataset
2. Cleans and preprocesses it
3. Splits into train/val/test
4. Trains both TF-IDF+SVM (SGD) and BERT models
5. Saves artifacts in experiments_<model>/ directories
"""

import os,sys
import pandas as pd
from pprint import pprint

# project root = /content/Assignment
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(PROJECT_ROOT)           # so relative paths (data/...) resolve from root
sys.path.insert(0, PROJECT_ROOT)

# -------------------------
# Local imports
# -------------------------
from src.preproecessing import clean_dataframe
from src.shufflesplit import shuffle_split
from models.tfidf_sgd import train_tfidf_sdg
from bert import train_bert_model


# -------------------------
# Paths & config
# -------------------------
DATA_PATH = "data/support_tickets_noisy.csv"
TARGET_COL = "category"

print("🚀 Starting training pipeline...")
print(f"Loading dataset from: {DATA_PATH}")

# -------------------------
# Step 1: Load dataset
# -------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

# -------------------------
# Step 2: Preprocess
# -------------------------
print("\n🧹 Cleaning and normalizing data...")
cleaned_df = clean_dataframe(df)
print(f"After cleaning: {len(cleaned_df)} rows")

# -------------------------
# Step 3: Split (Train/Val/Test)
# -------------------------
print("\n🔀 Splitting data into train/val/test sets...")
train_df, val_df, test_df = shuffle_split(cleaned_df)
print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

# -------------------------
# Step 4: Train TF-IDF + LinearSVC (SGD)
# -------------------------
print("\n⚙️ Training TF-IDF + LinearSVC model...")
sgd_model, sgd_summary = train_tfidf_sdg(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    target_col=TARGET_COL,
    upsample_minority=True,
    class_weight="balanced"
)
print("\n✅ TF-IDF + LinearSVC training complete.")
pprint(sgd_summary)

# -------------------------
# Step 5: Train BERT model
# -------------------------
print("\n🤖 Fine-tuning BERT model...")
from sklearn.utils import resample

def upsample_minority(df: pd.DataFrame, label_col: str, seed: int = 42,
                      target: str = "max", cap: int | None = None) -> pd.DataFrame:
    """
    target='max' -> upsample each class to the max class count.
    If cap is given, upsample to min(max_count, cap) to avoid huge blowups.
    """
    vc = df[label_col].value_counts()
    max_count = vc.max()
    target_n = min(max_count, cap) if (cap is not None) else max_count

    parts = []
    for label, group in df.groupby(label_col, group_keys=False):
        n = len(group)
        if n < target_n:
            need = target_n - n
            up = resample(group, replace=True, n_samples=need, random_state=seed)
            parts.append(pd.concat([group, up], ignore_index=True))
        else:
            # downsample if you want perfect balance to 'target_n'; otherwise keep as-is
            parts.append(group.sample(n=target_n, random_state=seed) if target == "max" and n > target_n else group)
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out

# 👉 Apply to train only
train_df_bal = upsample_minority(train_df, label_col="category", seed=42)
print("After upsample:", train_df_bal["category"].value_counts().to_dict())

# keep val_df and test_df untouched


bert_trainer, bert_summary = train_bert_model(
    train_df=train_df_bal,
    val_df=val_df,
    test_df=test_df,
    model_name="distilbert-base-uncased",
    target_col=TARGET_COL,
    batch_size=16,
    epochs=6,
    lr=2e-5,
    use_early_stopping=True
)
print("\n✅ BERT fine-tuning complete.")
pprint(bert_summary)

# -------------------------
# Final summary
# -------------------------
print("\n🏁 Training pipeline finished successfully.")
print("-" * 50)
print("TF-IDF + SVM summary:")
pprint(sgd_summary)
print("\nBERT summary:")
pprint(bert_summary)
print("-" * 50)
