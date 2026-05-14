# Teen Mental Health & Depression Prediction

A machine learning project to predict depression risk in teenagers
based on social media usage, sleep, stress, and lifestyle factors.

## Problem
Binary Classification — Predict `depression_label` (0 = No Depression, 1 = Depressed)

## Dataset
[Teenager Mental Health Dataset — Kaggle](https://www.kaggle.com/datasets/algozee/teenager-menthal-healy)  
1200 rows | 13 features

## Key Challenge
Severe class imbalance — only 31 depression cases out of 1200.  
Solved using **SMOTE** (Synthetic Minority Oversampling Technique).

## Approach
- EDA: depression patterns by gender, sleep, stress, social media
- Preprocessing: StandardScaler, OneHotEncoder, OrdinalEncoder
- Imbalance fix: SMOTE on training data only
- Model: Random Forest Classifier (100 estimators)
- Full sklearn + imbalanced-learn Pipeline

## Results
| Metric | Score |
|---|---|
| Accuracy | 98.7% |
| Precision (Depression) | 1.00 |
| Recall (Depression) | 0.50 |
| F1 Score (Depression) | 0.67 |

> Focus metric: **Recall** — missing a depressed teen is far worse than a false alarm.

## What I Learned
- High accuracy on imbalanced data is misleading
- SMOTE doubled recall from 0.17 → 0.50
- OrdinalEncoder for ordered categories vs OneHotEncoder for unordered

## How to Run
1. Clone repo
2. Place `Teen_Mental_Health_Dataset.csv` in root folder
3. `pip install imbalanced-learn scikit-learn pandas matplotlib seaborn`
4. Open `notebook.ipynb` and run all cells

## Tech Stack
`Python` `pandas` `scikit-learn` `imbalanced-learn` `matplotlib` `seaborn`
