# SaaS Support Ticket Classification

## 📘 Overview
This project automates classification of **SaaS customer support tickets** for categories such as:
- **Mobile App & UI Issues**
- **Times & Attendance**
- **Billing and Payments**
- **Technical Support**
- **Product Support**
- **IT Support**
- **Customer Service**
- **General Inquiry**
- **Human Resources**
- **Services Outages & Maintenance**

It provides a full ML workflow — from dataset creation, preprocessing, model training (baseline + BERT), evaluation, and inference — to FastAPI deployment.

---

## 📂 Deliverables

### 🗂️ Data Folder (`data/`)
The `data/` directory contains all datasets used and generated during experimentation.

#### 1️⃣ **dataset-tickets-multi-lang-4-20k.csv**
- **Source:** Public dataset sourced from **Kaggle**, containing multilingual SaaS support tickets.  
- **Purpose:** Used as a base reference for ticket language diversity and structure.  
- **Attributes:** Title, description, and category fields with multilingual text.

#### 2️⃣ **support_tickets_2000_with_dups.csv**
- **Source:** Generated via **prompt engineering using OpenAI’s GPT model**.  
- **Purpose:** Simulates realistic SaaS ticket data in English, including intentional duplicates, partial noise, and varied ticket lengths.  
- **Attributes:** Includes realistic user issues, support responses, and metadata (priority, type, category).

#### 3️⃣ **support_tickets_noisy.csv**
- **Final dataset** used for preprocessing and model training.  
- Created by **combining multilingual.csv and support_tickets_with_dups.csv**, then applying additional noise:
  - Random **typos** and **punctuation irregularities**
  - Injected **duplicate tickets**
  - Added mild text perturbations to increase model robustness  
- This blended dataset simulates a **production-like imbalance** scenario to test generalization and denoising capabilities.

**Sourcing & Bias Notes**
- All data is synthetic, modeled on typical SaaS support domains.
- Includes realistic text, varying ticket lengths, duplicates, and minor PII placeholders.

## 🧹 Preprocessing (`src/preprocessing.py`)
Key preprocessing pipeline:
1. **Redaction:** Replaces emails, phone numbers, IDs, MFA codes.
2. **Normalization:** Lowercasing, punctuation cleanup, whitespace trimming.
3. **Typo Correction:** Using `TextBlob` - performant sensitive
4. **Lemmatization:** With `spaCy` (English small model). - performant sensitive
5. **Stopword Removal:** Combines NLTK’s stopwords with custom words.
6. **Deduplication & Filtering:** Removes short&long description lengths or repeated tickets.

## 🔀 Split Strategy (`src/shufflescript.py`)
**Goal:** Prevent data leakage & mimic real-world ticket flow.

- Uses `GroupShuffleSplit` with `title + tag` as grouping key.
- Produces:
  - **70% Train**
  - **15% Validation**
  - **15% Test**
- Ensures no similar ticket appears across splits.
- Handles imbalance:
  - **Upsampling** minority classes.
  - **Class weights** during model training.

## 🤖 Models (`src/models/`)

### 1️⃣ Baseline: TF-IDF + SGD Classifier
**File:** `src/models/sdg_classifier.py`

- **Vectorizer:** Word 1–2 gram + Char 3–5 gram TF-IDF.
- **Classifier:** Classifier:** `LinearSVC` (Support Vector Machine) with optional `class_weight='balanced'`.
- **Training Strategy:**
- Handles imbalance via class weights & upsampling (Hybrid Approcah).
- Outputs:
  - `experiments_sgd/run-*/model.pkl`
  - `val_confusion.png`, `test_confusion.png`
  - `summary.json`

---

### 2️⃣ Main Model: BERT Fine-Tuning
**File:** `src/models/bert.py`

- Base model: `distilbert-base-uncased`
- Input text: Concatenated `title`, `description`, `answer`, `type`, `tag`
- Loss: Weighted CrossEntropy
- Handles imbalance via  Weighted CrossEntropy & upsampling (Hybrid Approcah).
- Outputs:
  - `experiments_bert/run-*/`
    - Model checkpoint (`pytorch_model.bin`)
    - `config.json`, `tokenizer.json`
    - Metrics and confusion matrix

---

## ⚙️ Training (`src/models/train.py`)
Handles end-to-end:
- Loads data
- Runs preprocessing
- Applies split strategy
- Trains both **SVC** and **BERT**
- Saves results to:
  ```
  experiments_bert/
  experiments_sgd/
  models/
  ```

Each run folder contains:
```
summary.json
val_confusion.png
test_confusion.png
model artifacts
```

## 📊 Evaluation & Error Analysis
Metrics computed per split:
- Accuracy
- Macro F1
- Precision & Recall
- Confusion Matrix

----

## 🤖 Why BERT Performs Better Than LinearSVC

Even though BERT and LinearSVC + TF-IDF achieved similar accuracy scores, BERT provides more reliable and meaningful predictions:

**Understands Context**:
BERT captures relationships between words (not just their frequency), so it handles reworded or paraphrased tickets more accurately.

**Handles Noise Better:**
Because it was pre-trained on large, diverse text corpora, BERT remains robust to typos, abbreviations, and punctuation errors in noisy support data.

**Higher Confidence & Stability:**
BERT outputs calibrated confidence scores (via softmax). Even when accuracy is similar, its predictions are more certain and consistent, as seen during inference.

**Generalization:**
BERT transfers well to unseen text or new ticket categories, while SVC models rely heavily on keywords seen during training.

In short:
BERT’s contextual understanding and stronger confidence calibration make it more production-ready and trustworthy than LinearSVC, even at similar numeric accuracy.

## 🔮 Inference (`src/infer.py`)
Command-line prediction interface:

```bash
python src/infer.py --model bert --model_dir experiments_bert/run-20251014-154000
```

Input JSON:
```json
 {"title": "Clock-in not working",
    "description": "App crashes when punching in."}
```

Output JSON:
```json
{"predicted_category": "Mobile App & UI issues", "confidence": 0.8837}
```

Low-confidence (< threshold) predictions are automatically labeled **"General Inquiry"**.

---

## 🌐 FastAPI Deployment (`src/app.py`)
Exposes a REST API with two endpoints:

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/health` | GET | Returns API and model status |
| `/classify` | POST | Classifies a new support ticket |

**Default Model:**  
Loads the **latest BERT run** from `experiments_bert/`.

--------

## 🧾 Outputs Summary
| Component | Deliverable | Location |
|------------|--------------|-----------|
| Dataset | Synthetic noisy tickets | `data/support_tickets_noisy.csv` |
| Preprocessing | Cleaning + redaction pipeline | `src/preprocessing.py` |
| Splitting | Grouped shuffle split | `src/shufflescript.py` |
| Models | Baseline (SGD) + Main (BERT) | `src/models/` |
| Training | Unified script | `src/models/train.py` |
| Evaluation | Metrics & confusion matrices | `experiments/` |
| Inference | CLI + API endpoints | `src/infer.py`, `src/app.py` |
| Documentation | End-to-end explanation | `README.md` |

---


## 💻 Running in Google Colab - Dayforce_Assignment.ipynb is uploaded to this folder, it contains all the steps from training to calling the API endpoints using ngrok

-------
## 🚀 Next Steps (Performance Improvements)
To increase accuracy and robustness:
- 🔁 **Generate more diverse synthetic data**
- 🔑 **Use API-based synthetic generation** (requires paid key)
- 🧠 **Fine-Tuning Paramters for BERT to increase performance** - With Increased Epochs, Increase Input Length(max_length=512),Tune Learning Rate & Scheduler, apply dropout and early stopping
- 🪄 **Reduce learning rate** for smoother convergence
- ⚡ **Fine-tune larger transformer models:**
  - `microsoft/MiniLM-L12-H384-uncased`
  - `roberta-base`, `roberta-large`
- 🧩 **Train on GPU (Colab Pro / A100)** for improved contextual learning.

-------

