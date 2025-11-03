#!/usr/bin/env python3
"""
Plot confusion matrix heatmap and feature importances.

Outputs:
 - reports/eval_confusion.png
 - reports/feature_importance.png
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Paths (change if you used different names)
DATA_CSV = "data/dataset.csv"
MODEL_PKL = "models/recommender_rf.pkl"
OUT_DIR = "reports"
CONF_PNG = os.path.join(OUT_DIR, "eval_confusion.png")
FI_PNG = os.path.join(OUT_DIR, "feature_importance.png")

os.makedirs(OUT_DIR, exist_ok=True)

# Load data and model
df = pd.read_csv(DATA_CSV)
X = df[['cpu_peak_percent','mem_95_mb','cpu_workers','duration_s']]
y = df['recommended_instance']

clf = joblib.load(MODEL_PKL)

# Cross-validated predictions (same CV setup used earlier)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
y_pred = cross_val_predict(clf, X, y, cv=cv)

# Confusion matrix (counts)
labels = np.unique(y)  # consistent class order
cm = confusion_matrix(y, y_pred, labels=labels)

# Plot confusion matrix as heatmap (counts + percentages)
fig, ax = plt.subplots(figsize=(6,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
ax.set_title("Confusion Matrix (3-fold CV)")
plt.tight_layout()
plt.savefig(CONF_PNG, dpi=200)
plt.close()

# Also save a normalized version (percentage) beside it for clarity
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_title("Confusion Matrix (Normalized by True Class)")
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)
# annotate percentages
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        text = f"{cm[i,j]} / {cm_norm[i,j]*100:.1f}%"
        ax.text(j, i, text, ha="center", va="center", color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
norm_png = os.path.join(OUT_DIR, "eval_confusion_normalized.png")
plt.savefig(norm_png, dpi=200)
plt.close()

# Feature importances (from the trained model)
try:
    importances = clf.feature_importances_
    feature_names = X.columns.tolist()
    # Sort by importance
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.bar(np.arange(len(importances)), importances[order], tick_label=[feature_names[i] for i in order])
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importances (trained model)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FI_PNG, dpi=200)
    plt.close()
except Exception as e:
    print("Could not compute feature importances:", e)

print("Saved:")
print(" -", CONF_PNG)
print(" -", norm_png)
print(" -", FI_PNG)
