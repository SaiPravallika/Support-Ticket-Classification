#!/usr/bin/env python3
"""
infer.py
--------
CLI for inference with either:
  - TF-IDF + LinearSVC (SGD) pipeline saved via joblib/pickle
  - Fine-tuned BERT/HF model folder (transformers)

Usage examples:
  python infer.py --model sgd --model_dir experiments_sgd/run-XXXX/model.pkl \
      --input_json '{"title":"App crash","description":"Crashes on punch in","answer":"","type":"Incident","tag":"Crash"}'

  python infer.py --model bert --model_dir experiments_bert/run-XXXX \
      --threshold 0.55 \
      --input_json '{"title":"Login loop","description":"Stuck after MFA","answer":"", "type":"Incident","tag":"Login"}'

Input JSON (fields are optional except title/description):
{
  "title": "...",
  "description": "...",
  "answer": "",
  "type": "",
  "tag": ""
}

Output:
{
  "predicted_category": "Mobile App & UI issues",
  "confidence": 0.82,
  "raw": {"scores": [...], "labels": [...]}
}
"""

import os
import io
import json
import argparse
import numpy as np

# ----------------------------
# Helpers
# ----------------------------

def _safe_str(x):
    return "" if x is None else str(x)

def fuse_for_sgd(title, description):
    # Match training for tfidf_sdg.py (title + description)
    return f"{_safe_str(title)} {_safe_str(description)}".strip()

def fuse_for_bert(title, description, answer="", typ="", tag=""):
    # Match training for bert_trainer.py
    return " ".join([
        "Title:", _safe_str(title),
        "[SEP] Description:", _safe_str(description),
        "[SEP] Answer:", _safe_str(answer),
        "[SEP] Type:", _safe_str(typ),
        "[SEP] Tag:", _safe_str(tag),
    ]).strip()

def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    s = e / (np.sum(e) + 1e-12)
    return s

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def json_or_path_to_dict(s: str):
    if os.path.exists(s):
        with open(s, "r") as f:
            return json.load(f)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        raise ValueError("Provide valid JSON via --input_json or a path to a JSON file via --input_file.")

# ----------------------------
# SGD (TF-IDF + LinearSVC) loader & predictor
# ----------------------------

def load_sgd(model_path: str):
    import pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    # classes_ lives on final estimator; try best-effort access
    classes = getattr(getattr(model, "classes_", None), "tolist", lambda: None)()
    if classes is None:
        # sometimes classes_ is on inner calibrated classifier; try digging
        try:
            classes = model.named_steps["clf"].classes_.tolist()
        except Exception:
            classes = None
    return model, classes

def predict_sgd(model, classes, payload, threshold: float):
    text = fuse_for_sgd(payload.get("title"), payload.get("description"))

    # Confidence:
    # 1) If calibrated, use predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        idx = int(np.argmax(proba))
        conf = float(np.max(proba))
        label = classes[idx] if classes is not None else model.classes_[idx]
        return label, conf, proba.tolist(), (classes if classes is not None else model.classes_.tolist())

    # 2) Else, approximate from decision_function
    if hasattr(model, "decision_function"):
        scores = model.decision_function([text])[0]
        # Binary case → 1D margin
        if np.ndim(scores) == 0 or (isinstance(scores, (float, np.floating))):
            # Convert margin to probability with sigmoid
            p1 = float(sigmoid(scores))
            proba = np.array([1.0 - p1, p1], dtype=np.float32)
            idx = int(np.argmax(proba))
            labels = classes if classes is not None else model.classes_.tolist()
            label = labels[idx]
            conf = float(np.max(proba))
            return label, conf, proba.tolist(), labels
        else:
            # Multiclass: softmax over margins
            proba = softmax(scores)
            idx = int(np.argmax(proba))
            labels = classes if classes is not None else model.classes_.tolist()
            label = labels[idx]
            conf = float(np.max(proba))
            return label, conf, proba.tolist(), labels

    # 3) Fallback: no score; just predict
    pred = model.predict([text])[0]
    return pred, 0.5, [], classes if classes is not None else []

# ----------------------------
# BERT loader & predictor
# ----------------------------

def load_bert(model_dir: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    # Load labels if present
    label2id_path = os.path.join(model_dir, "label2id.json")
    id2label_path = os.path.join(model_dir, "id2label.json")
    if os.path.exists(id2label_path):
        with open(id2label_path, "r") as f:
            id2label = json.load(f)
        # keys as strings → ensure index order
        num_labels = model.config.num_labels
        labels = [id2label[str(i)] if str(i) in id2label else str(i) for i in range(num_labels)]
    else:
        # fallback to model config
        num_labels = model.config.num_labels
        if hasattr(model.config, "id2label") and model.config.id2label:
            labels = [model.config.id2label[i] for i in range(num_labels)]
        else:
            labels = [str(i) for i in range(num_labels)]
    device = "cuda" if hasattr(model, "device") and str(model.device).startswith("cuda") else "cpu"
    return tokenizer, model, labels

def predict_bert(tokenizer, model, labels, payload, threshold: float):
    import torch
    text = fuse_for_bert(
        payload.get("title"),
        payload.get("description"),
        payload.get("answer", ""),
        payload.get("type", ""),
        payload.get("tag", "")
    )
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=384, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]
    proba = softmax(logits)
    idx = int(np.argmax(proba))
    label = labels[idx]
    conf = float(np.max(proba))
    return label, conf, proba.tolist(), labels

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sgd", "bert"], required=True, help="Which model type to load.")
    ap.add_argument("--model_dir", required=True,
                    help="Path to model: for sgd → path to model.pkl; for bert → path to HF folder.")
    ap.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold; below → 'General'.")
    # Input (one of the following)
    ap.add_argument("--input_json", type=str, default=None, help="Inline JSON string or path to JSON file.")
    ap.add_argument("--input_file", type=str, default=None, help="Path to JSON file (alternative to --input_json).")
    args = ap.parse_args()

    # Load payload
    if args.input_json:
        payload = json_or_path_to_dict(args.input_json)
    elif args.input_file:
        with open(args.input_file, "r") as f:
            payload = json.load(f)
    else:
        # read stdin if nothing provided
        payload = json.load(io.TextIOWrapper(buffer=None, encoding="utf-8", errors="ignore"))

    # Minimal validation
    if not payload.get("title") or not payload.get("description"):
        raise ValueError("Input JSON must include at least 'title' and 'description'.")

    # Predict
    if args.model == "sgd":
        model, classes = load_sgd(args.model_dir)
        label, conf, scores, label_list = predict_sgd(model, classes, payload, args.threshold)
    else:
        tokenizer, model, labels = load_bert(args.model_dir)
        label, conf, scores, label_list = predict_bert(tokenizer, model, labels, payload, args.threshold)

    # Thresholding → "General"
    final_label = "General" if conf < args.threshold else label

    out = {
        "predicted_category": final_label,
        "confidence": round(conf, 6),
        "raw": {
            "scores": scores,
            "labels": label_list
        }
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
