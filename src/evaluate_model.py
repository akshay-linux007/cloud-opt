#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Load dataset
df = pd.read_csv('data/dataset.csv')
X = df[['cpu_peak_percent','mem_95_mb','cpu_workers','duration_s']]
y = df['recommended_instance']

# Load trained model
clf = joblib.load('models/recommender_rf.pkl')

# Cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
preds = cross_val_predict(clf, X, y, cv=cv)

print("Classification report (3-fold CV):")
print(classification_report(y, preds))

print("Confusion matrix (3-fold CV):")
print(confusion_matrix(y, preds))

# Feature importances
print("\nFeature importances from trained model:")
for name, imp in zip(X.columns, clf.feature_importances_):
    print(f"{name}: {imp:.4f}")
