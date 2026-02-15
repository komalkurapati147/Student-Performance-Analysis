# Student Performance Prediction using Machine Learning

## Project Overview

Built a supervised machine learning classification system using the Student Performance Data Set to predict whether a student will pass or fail based on academic and behavioral attributes.

Achieved 92.4% test accuracy using Random Forest and 91.76% cross-validation score, demonstrating strong generalization performance.

---

## Objective

To develop a predictive model that:

- Identifies at-risk students early
- Compares multiple classification algorithms
- Optimizes performance using cross-validation and hyperparameter tuning
- Determines key academic factors influencing outcomes

---

## Dataset Details

- Dataset: Student Performance Data Set
- Total Records: 395 students
- Total Features: 33 attributes
- Problem Type: Binary Classification

### Target Engineering

Converted final grade (G3) into binary outcome:

- Pass (1): G3 >= 10
- Fail (0): G3 < 10

Created new target column: `Performance`.

---

## Data Preprocessing & Feature Engineering

- Renamed relevant columns for clarity
- Encoded categorical variable (Gender → 0/1)
- Selected 7 key predictive features
- Applied StandardScaler for normalization
- Performed stratified 80/20 train-test split to preserve class distribution
- Used random_state=42 for reproducibility

### Features Used

- Gender
- Age
- StudyHours
- PreviousFailures
- Absences
- G1 (First Period Grade)
- G2 (Second Period Grade)

---

## Models Implemented

1. Logistic Regression  
2. Support Vector Machine (SVM)  
3. Random Forest Classifier  
4. Hyperparameter-Tuned Random Forest (GridSearchCV, 5-fold CV)

Hyperparameters tuned:
- n_estimators: [100, 200]
- max_depth: [None, 5, 10]

---

## Model Performance Comparison

| Model                   | Test Accuracy |
|--------------------------|---------------|
| Logistic Regression      | 88.6%         |
| SVM                      | 87.3%         |
| Random Forest            | 92.4%         |
| Tuned Random Forest      | 87.3%         |
| 5-Fold CV (Random Forest)| 91.76%        |

Random Forest achieved the highest performance with strong cross-validation stability, indicating low overfitting and reliable generalization.

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve (AUC)
- Feature Importance Analysis

The model achieved high recall for Pass classification while maintaining strong overall precision balance.

---

## Key Insights

- G1 and G2 (previous grades) are the strongest predictors of final performance.
- PreviousFailures significantly increases probability of failure.
- Ensemble methods outperform linear models on this dataset.
- Cross-validation confirms model robustness and stability.

---

## Business Impact

This model can:

- Enable early identification of academically at-risk students
- Support targeted intervention strategies
- Improve institutional pass rates
- Promote data-driven academic planning

---

## Skills Demonstrated

- End-to-end Machine Learning pipeline
- Data preprocessing and feature engineering
- Stratified sampling and scaling
- Model comparison and evaluation
- Hyperparameter tuning using GridSearchCV
- Cross-validation techniques
- Performance visualization
- Analytical reasoning and model interpretation

---

## How to Run

1. Install dependencies:

   pip install pandas numpy matplotlib seaborn scikit-learn

2. Ensure `student-mat.csv` is in the same directory.

3. Run:

   python your_script_name.py




