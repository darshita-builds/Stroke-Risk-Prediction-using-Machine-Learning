# Stroke-Risk-Prediction-using-Machine-Learning

# Stroke Risk Prediction using Machine Learning

A comparative study of four classification algorithms — **Logistic Regression**, **K-Nearest Neighbors**, **Decision Tree**, and **SVM (RBF kernel)** — for predicting stroke risk from real patient health records.

## Overview

Stroke is one of the leading causes of death and long-term disability worldwide, and early identification of at-risk patients is critical for timely intervention. This project builds a complete, end-to-end supervised-learning pipeline on a real clinical dataset: exploratory data analysis → statistical hypothesis testing → outlier removal → model training → evaluation → best-model selection, with class imbalance handled via SMOTE.

## Dataset

- **Source:** Healthcare Stroke Prediction Dataset (real, publicly documented clinical records)
- **Size:** 5,110 patients, 12 attributes (demographic, lifestyle, and clinical)
- **Target:** `stroke` (binary) — only 249 positive cases (~4.9%), a realistic and heavily imbalanced target
- `healthcare-dataset-stroke-data.csv` is included in this repo so the notebook runs standalone.

## Methodology

1. **Data cleaning** — imputed 201 missing BMI values with the median; dropped 1 record with an unspecified gender category.
2. **Hypothesis testing** — tested whether age, hypertension, and average glucose level are significantly associated with stroke (Welch's t-test / chi-square). All three null hypotheses were rejected (p < 0.05).
3. **Outlier removal** — IQR-based filtering on `avg_glucose_level` and `bmi` (5,109 → 4,390 rows).
4. **Preprocessing** — `StandardScaler` on numeric features, one-hot encoding on categorical features, stratified 80/20 train/test split.
5. **Class imbalance handling** — SMOTE applied to the training set only (test set kept at its original, real-world distribution).
6. **Modeling** — trained and compared 4 classifiers: Logistic Regression, KNN (k=7), Decision Tree (max_depth=6), SVM (RBF kernel).
7. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, and a full confusion matrix (TP / TN / FP / FN) for each model.

## Results

| Model | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 21 | 632 | 213 | 12 | 0.744 | 0.090 | **0.636** | 0.157 | **0.784** |
| Decision Tree | 19 | 672 | 173 | 14 | 0.787 | 0.099 | 0.576 | 0.169 | 0.760 |
| SVM (RBF Kernel) | 17 | 677 | 168 | 16 | 0.790 | 0.092 | 0.515 | 0.156 | 0.749 |
| K-Nearest Neighbors | 13 | 692 | 153 | 20 | 0.803 | 0.078 | 0.394 | 0.131 | 0.677 |

**Best model: Logistic Regression** — highest ROC-AUC and the strongest recall/precision balance. In a medical screening context, recall is prioritized over raw accuracy, since a missed stroke case (false negative) is far costlier than a false alarm.

> **Note on precision:** precision is low across all four models (~8–10%). This isn't a bug — it's the direct, expected consequence of predicting a rare event (~5% positive rate) with general-purpose classifiers, and is called out here deliberately rather than hidden. It's also why accuracy alone is a misleading metric for this problem: a model that always predicts "no stroke" would score ~95% accuracy while being clinically useless.

## Key Insights

- **Age** and **average glucose level** were the two strongest predictors across every model and every hypothesis test, consistent with established clinical risk factors.
- Class imbalance is the central challenge in this dataset — model selection was driven by ROC-AUC and
