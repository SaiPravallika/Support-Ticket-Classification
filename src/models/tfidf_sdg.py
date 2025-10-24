"""
models/tfidf_sdg.py
-------------------
Defines the TF-IDF + LinearSVC training pipeline.
Handles class imbalance (optional upsampling), evaluates, and saves artifacts.
"""

import os
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from utils import make_run_dir


# ---------------------------------------------------
# Helper utilities
# ---------------------------------------------------

def upsample_by_class(df, label_col: str, seed: int = 42) -> pd.DataFrame:
    """Upsample minority classes to match the majority count."""
    max_count = df[label_col].value_counts().max()
    parts = []
    for label, group in df.groupby(label_col):
        if len(group) < max_count:
            up = resample(group, replace=True, n_samples=max_count - len(group), random_state=seed)
            parts.append(pd.concat([group, up], ignore_index=True))
        else:
            parts.append(group)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def plot_confusion(cm, labels, path: Path):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def evaluate(model, X, y, split_name: str):
    """Evaluate trained model on validation/test set."""
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    macro = f1_score(y, pred, average="macro")
    print(f"\n=== {split_name.upper()} ===")
    print(f"accuracy={acc:.4f}  macro_f1={macro:.4f}")
    print(classification_report(y, pred, zero_division=0))
    cm = confusion_matrix(y, pred).tolist()
    return {"accuracy": acc, "macro_f1": macro, "cm": cm}


# ---------------------------------------------------
# Main training entry
# ---------------------------------------------------

def train_tfidf_sdg(train_df, val_df, test_df,
                    target_col: str = "category",
                    upsample_minority: bool = True,
                    class_weight: str | None = "balanced",
                    seed: int = 42):
    """
    Train a TF-IDF + LinearSVC classifier and save run artifacts.

    Args:
        train_df, val_df, test_df (pd.DataFrame): Data splits.
        target_col (str): Column to predict ('category', 'type', etc.).
        upsample_minority (bool): Whether to balance training data.
        class_weight (str|None): LinearSVC class weighting.
        seed (int): Random seed.

    Returns:
        (Pipeline, dict): Trained model and summary metrics.
    """

    assert all(col in train_df.columns for col in ["title", "description", target_col]), \
        "train_df must include 'title', 'description', and the target column."

    # Upsample minority classes if enabled
    if upsample_minority:
        train_df = upsample_by_class(train_df, target_col, seed)
        print(f"✅ Upsampled training data to balance classes (seed={seed})")
    else:
        print("⚠️ No upsampling applied (using natural class distribution)")

    # TF-IDF features (word + char)
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        lowercase=True
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        lowercase=True
    )
    feats = FeatureUnion([("w", word_vec), ("c", char_vec)])

    clf = LinearSVC(C=1.0, class_weight=class_weight, random_state=seed)
    model = Pipeline([("feats", feats), ("clf", clf)])

    # Training data
    X_train = (train_df["title"] + " " + train_df["description"]).astype(str)
    y_train = train_df[target_col].astype(str)
    X_val = (val_df["title"] + " " + val_df["description"]).astype(str)
    y_val = val_df[target_col].astype(str)
    X_test = (test_df["title"] + " " + test_df["description"]).astype(str)
    y_test = test_df[target_col].astype(str)

    print("\n🚀 Training LinearSVC (TF-IDF)...")
    model.fit(X_train, y_train)

    # Evaluate
    val_metrics = evaluate(model, X_val, y_val, "val")
    test_metrics = evaluate(model, X_test, y_test, "test")

    # Save artifacts
    run_dir = make_run_dir("sgd")
    run_path = Path(run_dir)
    labels = sorted(list(set(y_train) | set(y_val) | set(y_test)))

    with open(run_path / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    plot_confusion(val_metrics["cm"], labels, run_path / "val_confusion.png")
    plot_confusion(test_metrics["cm"], labels, run_path / "test_confusion.png")

    summary = {
        "target": target_col,
        "upsample": upsample_minority,
        "class_weight": class_weight,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
    }

    with open(run_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Run completed. Artifacts saved to: {run_path.resolve()}")
    return model, summary
