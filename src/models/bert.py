"""
models/bert_trainer.py
----------------------
Defines a reusable BERT fine-tuning pipeline with weighted loss,
evaluation, early stopping, and artifact saving.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from utils import make_run_dir


# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------

def _to_hf(df: pd.DataFrame, label2id: dict) -> Dataset:
    """Convert a pandas DataFrame to a Hugging Face Dataset."""
    df = df.copy()
    df["text"] = (
        "Title: " + df["title"].fillna("") + " [SEP] " +
        "Description: " + df["description"].fillna("") + " [SEP] " +
        "Answer: " + df.get("answer", "").fillna("") + " [SEP] " +
        "Type: " + df.get("type", "").fillna("") + " [SEP] " +
        "Tag: " + df.get("tag", "").fillna("")
    )
    df["label"] = df["category"].astype(str).map(label2id)
    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)


def _compute_metrics(eval_pred):
    """Compute accuracy and macro-F1 for HuggingFace Trainer."""
    logits, y_true = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(y_true, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1, "macro_precision": p, "macro_recall": r}


class WeightedTrainer(Trainer):
    """Custom Trainer with class-weighted loss."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels_t = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels_t.view(-1))
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------
# Main training entry
# ---------------------------------------------------

def train_bert_model(
    train_df,
    val_df,
    test_df,
    model_name: str = "distilbert-base-uncased",
    target_col: str = "category",
    batch_size: int = 16,
    epochs: int = 10,
    lr: float = 2e-5,
    seed: int = 42,
    use_early_stopping: bool = True,
):
    """
    Fine-tune a transformer (BERT/DistilBERT/etc.) on ticket classification.

    Args:
        train_df, val_df, test_df (pd.DataFrame): Dataset splits.
        model_name (str): Pretrained model checkpoint.
        target_col (str): Target column (usually 'category').
        batch_size (int): Per-device batch size.
        epochs (int): Max training epochs.
        lr (float): Learning rate.
        seed (int): Random seed.
        use_early_stopping (bool): Whether to enable early stopping.

    Returns:
        trainer, summary (dict)
    """

    # ---------------------------------------------------
    # Label setup
    # ---------------------------------------------------
    labels = sorted(train_df[target_col].astype(str).unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    # ---------------------------------------------------
    # Tokenization
    # ---------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    hf_train = _to_hf(train_df, label2id)
    hf_val = _to_hf(val_df, label2id)
    hf_test = _to_hf(test_df, label2id)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=384)

    hf_train = hf_train.map(tok, batched=True).remove_columns(["text"])
    hf_val = hf_val.map(tok, batched=True).remove_columns(["text"])
    hf_test = hf_test.map(tok, batched=True).remove_columns(["text"])

    hf_train.set_format("torch")
    hf_val.set_format("torch")
    hf_test.set_format("torch")

    hf_ds = DatasetDict(train=hf_train, validation=hf_val, test=hf_test)

    # ---------------------------------------------------
    # Compute class weights
    # ---------------------------------------------------
    train_labels_np = np.array(train_df[target_col].map(label2id).tolist())
    class_counts = np.bincount(train_labels_np, minlength=len(labels))
    weights = (class_counts.sum() / np.maximum(class_counts, 1)).astype(np.float32)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights)

    # ---------------------------------------------------
    # Model & arguments
    # ---------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    model.config.hidden_dropout_prob = 0.3
    model.config.attention_probs_dropout_prob = 0.3

    run_dir = make_run_dir("bert")


    args = TrainingArguments(
        output_dir=run_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=True,
        report_to=[]
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if use_early_stopping else []

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=hf_ds["train"],
        eval_dataset=hf_ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
        class_weights=class_weights,
        callbacks=callbacks
    )

    # ---------------------------------------------------
    # Train + evaluate
    # ---------------------------------------------------
    trainer.train()

    test_out = trainer.predict(hf_ds["test"])
    test_metrics = test_out.metrics
    y_true = test_out.label_ids
    y_pred = np.argmax(test_out.predictions, axis=-1)
    cm = confusion_matrix(y_true, y_pred).tolist()

    labels_order = [id2label[i] for i in range(len(labels))]

    # ---------------------------------------------------
    # Save artifacts
    # ---------------------------------------------------
    with open(os.path.join(run_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)
    with open(os.path.join(run_dir, "id2label.json"), "w") as f:
        json.dump(id2label, f, indent=2)
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump({**test_metrics, "labels": labels_order, "cm": cm}, f, indent=2)

    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)

    # Confusion matrix visualization
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels_order)))
    ax.set_yticks(range(len(labels_order)))
    ax.set_xticklabels(labels_order, rotation=45, ha="right")
    ax.set_yticklabels(labels_order)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels_order)):
        for j in range(len(labels_order)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "test_confusion.png"), dpi=160)
    plt.close(fig)

    print(f"\n✅ BERT fine-tuning complete. Artifacts saved to: {run_dir}")

    summary = {
        "run_dir": run_dir,
        "model_name": model_name,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "test_accuracy": float(test_metrics.get("test_accuracy", 0.0)),
        "test_macro_f1": float(test_metrics.get("test_macro_f1", 0.0))
    }

    return trainer, summary
