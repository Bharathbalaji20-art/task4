import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

print("Files in /content directory:", os.listdir('/content'))
df = pd.read_csv("/content/data.csv")

df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

if df['diagnosis'].dtype == 'object':
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])

X = df.drop(columns='diagnosis')
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

conf_mat = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("Confusion Matrix:\n", conf_mat)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

threshold = 0.6
y_tuned_pred = (y_proba >= threshold).astype(int)

print(f"\nAfter tuning threshold to {threshold}:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_tuned_pred))
print(f"Precision: {precision_score(y_test, y_tuned_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_tuned_pred):.2f}")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x_vals = np.linspace(-10, 10, 200)
y_vals = sigmoid(x_vals)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label='Sigmoid Curve')
plt.axvline(0, color='red', linestyle='--', label='Threshold at 0')
plt.title("Sigmoid Function")
plt.xlabel("Input z")
plt.ylabel("Predicted Probability")
plt.grid(True)
plt.legend()
plt.show()
