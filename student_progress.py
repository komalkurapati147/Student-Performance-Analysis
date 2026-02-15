import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ==============================
# Load Data
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "student-mat.csv")
data = pd.read_csv(file_path, sep=';')

print("Dataset Shape:", data.shape)

# ==============================
# Data Preparation
# ==============================

# Rename columns for clarity
data.rename(columns={
    'sex': 'Gender',
    'studytime': 'StudyHours',
    'failures': 'PreviousFailures',
    'G3': 'FinalGrade'
}, inplace=True)

# Convert Gender to numeric
data['Gender'] = data['Gender'].map({'F': 0, 'M': 1})

# Create binary Performance column (Pass=1, Fail=0)
data['Performance'] = data['FinalGrade'].apply(lambda x: 1 if x >= 10 else 0)

# ==============================
# Feature Selection (Improved)
# ==============================
X = data[
    ['Gender',
     'age',
     'StudyHours',
     'PreviousFailures',
     'absences',
     'G1',
     'G2']
]

y = data['Performance']

# ==============================
# Train Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ==============================
# Scaling
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# Model 1: Logistic Regression
# ==============================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)

# ==============================
# Model 2: SVM
# ==============================
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

# ==============================
# Model 3: Random Forest
# ==============================
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# ==============================
# Cross Validation (RF)
# ==============================
rf_cv = cross_val_score(rf_model, X_train, y_train, cv=5).mean()

# ==============================
# Hyperparameter Tuning (RF)
# ==============================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
best_pred = best_rf.predict(X_test)
best_acc = accuracy_score(y_test, best_pred)

# ==============================
# Model Comparison
# ==============================
print("\n====== Model Comparison ======")
print("Logistic Regression Accuracy:", log_acc)
print("SVM Accuracy:", svm_acc)
print("Random Forest Accuracy:", rf_acc)
print("Tuned Random Forest Accuracy:", best_acc)
print("Cross Validation RF Score:", rf_cv)

print("\nBest Parameters:", grid.best_params_)

print("\nClassification Report (Best RF):\n")
print(classification_report(y_test, best_pred))

# ==============================
# Confusion Matrix
# ==============================
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Tuned RF")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# ROC Curve
# ==============================
y_prob = best_rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tuned RF")
plt.legend()
plt.show()

# ==============================
# Feature Importance
# ==============================
importances = best_rf.feature_importances_

plt.figure(figsize=(6,4))
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importance (Tuned RF)")
plt.show()




