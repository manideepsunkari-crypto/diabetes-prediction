# Diabetes Prediction — End-to-End ML Study

A supervised learning project that goes beyond accuracy — built to understand which model assumptions hold up, where they break, and why evaluation metrics matter more than a single number.

---

## The question

Can early-stage diabetes be reliably detected from clinical features alone? And if so, which model works better — and *why*?

---

## Tech Stack

- **Language:** Python
- **Libraries:** scikit-learn, pandas, NumPy, matplotlib, seaborn
- **Methods:** Logistic Regression, Decision Trees, model evaluation, ablation-style analysis

---

## Dataset

Used the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — 768 samples, 8 clinical features, binary classification (diabetic / non-diabetic).

---

## What I actually did

### 1. Data cleaning & EDA
- Several features had biologically impossible zero values (e.g. blood pressure = 0) — replaced with column medians
- Checked class imbalance: ~35% positive cases
- Plotted feature distributions to understand what the model would actually be working with

### 2. Model training
Trained two classifiers and compared them directly:
- **Logistic Regression** — assumes a linear decision boundary; interpretable coefficients
- **Decision Tree** — non-linear, can capture interactions but prone to overfitting

### 3. Ablation analysis
Not just "which model is better" but *why*:
- Removed features one at a time to measure individual contribution
- Varied tree depth to observe the overfitting curve
- Compared performance on training vs test set to check for generalisation

### 4. Evaluation — beyond accuracy
Accuracy alone is misleading on imbalanced medical data. Evaluated using:

| Metric | Why it matters here |
|--------|-------------------|
| Precision | Of predicted diabetics, how many actually are? |
| Recall | Of actual diabetics, how many did we catch? |
| F1-score | Balance between precision and recall |
| ROC-AUC | Model's ability to discriminate across thresholds |

**Best result: 90%+ accuracy, with strong recall** — prioritised catching true positives given the cost of a missed diagnosis.

---

## Results Summary

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | ~78% | ~0.74 | ~0.83 |
| Decision Tree | ~90%+ | ~0.88 | ~0.91 |

---

## Key findings

- Glucose level was the single most predictive feature across both models
- Decision tree outperformed logistic regression — the relationship between features and outcome is non-linear
- Deeper trees overfit noticeably; optimal depth was found through cross-validation
- Recall improved significantly after handling zero-value imputation — data cleaning had a measurable impact on results

---

## Limitations

- Dataset is relatively small (768 samples) — results may not generalise broadly
- No hyperparameter tuning beyond tree depth
- Did not test ensemble methods (Random Forest, XGBoost) — a clear next step
- Class imbalance was noted but not explicitly handled (e.g. SMOTE) — worth exploring

---

## How to Run

```bash
git clone https://github.com/yourusername/diabetes-prediction
cd diabetes-prediction
pip install -r requirements.txt
jupyter notebook diabetes_prediction.ipynb
```

---

## What I learned

- Data cleaning is not a formality — it directly changed model performance in this project
- Choosing the right evaluation metric matters more than optimising a single number
- Ablation-style thinking (remove one thing, measure the effect) is a cleaner way to understand a model than just reading feature importances
