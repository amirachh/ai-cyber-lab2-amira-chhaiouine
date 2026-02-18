# AI Cyber Lab 2 — Phishing URL Detection

## 1. Project Description
This project trains and evaluates a **binary phishing detector** using URL- and webpage-derived features.  
The pipeline:
- loads a tabular dataset,
- identifies and normalizes the target label,
- trains a Logistic Regression baseline model,
- evaluates classification performance,
- saves metrics and a confusion matrix artifact.

The repository is designed as a simple, reproducible baseline for cybersecurity ML experimentation.

## 2. Dataset Source and Features
- **Primary source:** Kaggle — **Phishing Detection** dataset by shashwatwork.
- **Dataset link:** https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset?resource=download
- **Dataset file in this repo:** `dataset.csv` (local copy used for this project).
- **Model input format:** numerical and boolean feature columns (non-numeric columns are dropped during preprocessing).
- **Target column:** automatically inferred from common names (for example `label`, `class`, `target`, `status`, `phishing`).

### Feature overview
The dataset contains URL lexical and host/page behavior signals, for example:
- URL structure counts (`length_url`, `nb_dots`, `nb_slash`, `nb_qm`, etc.)
- Host/domain indicators (`length_hostname`, `ip`, `domain_age`, `domain_registration_length`)
- Content/link behavior (`nb_hyperlinks`, `ratio_extHyperlinks`, `iframe`, `popup_window`, `safe_anchor`)
- Reputation/index features (`google_index`, `page_rank`, `web_traffic`)

The label is converted to binary (`0` = legitimate, `1` = phishing) before training.

## 3. Installation Instructions
```bash
# 1) Clone repository
git clone <your-repo-url>
cd ai-cyber-lab2-amira-chhaiouine

# 2) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt
```

### Prepare dataset path expected by the training/evaluation scripts
The scripts read from `data/processed/dataset.csv` by default.

```bash
mkdir -p data/processed
cp dataset.csv data/processed/dataset.csv
```

## 4. Training and Evaluation Commands
```bash
# Train Logistic Regression model and save it to results/model.joblib
python -m src.train

# Evaluate on test split and save metrics + confusion matrix
python -m src.eval
```

Generated artifacts:
- `results/model.joblib`
- `results/metrics.json`
- `results/confusion_matrix.png`

## 5. Baseline Results
Baseline metrics currently stored in `results/metrics.json`:

- **Accuracy:** 0.8141
- **Precision:** 0.8050
- **Recall:** 0.8229
- **F1-score:** 0.8138

These numbers come from the Logistic Regression baseline (`max_iter=1000`, `class_weight="balanced"`) on an 80/20 train-test split.

## 6. Ethics and Safety Considerations
- **False positives and false negatives:** Misclassification can block legitimate websites or miss real phishing pages. This model should assist, not replace, defense-in-depth controls.
- **Adversarial adaptation:** Attackers can modify URL patterns and content behavior over time; periodic retraining and monitoring are required.
- **Dataset bias and drift:** Performance depends on how representative the dataset is across regions, languages, and attack families.
- **Privacy and compliance:** If integrating with production telemetry, ensure legal basis, data minimization, and secure retention practices.
- **Responsible deployment:** Use confidence thresholds, human review for high-impact actions, and logging/auditing to reduce harm.

---

## Repository Structure
```text
src/data.py     # data loading and preprocessing
src/train.py    # model training script
src/eval.py     # evaluation + artifact generation
results/        # saved outputs (metrics, confusion matrix, model)
```
