import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ---------------------------------------
# 🔹 Set Output Directory
# ---------------------------------------
output_dir = "D:/ENT_DISEASE_DETECTION/logistic_regression"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------
# 🔹 Load Test Features and Labels
# ---------------------------------------
test_features = np.load("D:/ENT_DISEASE_DETECTION/test_features.npy")
test_labels = np.load("D:/ENT_DISEASE_DETECTION/test_labels.npy")

# ---------------------------------------
# 🔹 Load Logistic Regression Model and Scaler
# ---------------------------------------
log_reg_model = joblib.load("D:/ENT_DISEASE_DETECTION/ent_disease_logreg_model.pkl")
scaler = joblib.load("D:/ENT_DISEASE_DETECTION/logreg_scaler.pkl")

# Normalize test features
test_features = scaler.transform(test_features)

# Predict on test set
y_pred = log_reg_model.predict(test_features)
y_pred_probs = log_reg_model.predict_proba(test_features)  # Get probability scores for ROC Curve

# Compute accuracy
accuracy = accuracy_score(test_labels, y_pred)
print(f"\n✅ Logistic Regression Model Accuracy: {accuracy * 100:.2f}%")

# Classification Report as a dictionary
class_report = classification_report(test_labels, y_pred, output_dict=True)

# Extract Precision, Recall, and F1-score
labels = ["Chronic Otitis", "Myringosclerosis", "Normal Ear", "Ear Wax"]
metrics = ["Precision", "Recall", "F1-Score"]
metrics_matrix = np.array([[class_report[str(i)][metric.lower()] for metric in metrics] for i in range(len(labels))])

# Compute Confusion Matrix
conf_matrix = confusion_matrix(test_labels, y_pred)

# ---------------------------------------
# 🔹 Save Paths for Plots
# ---------------------------------------
conf_matrix_path = os.path.join(output_dir, "logreg_confusion_matrix.jpg")
class_dist_path = os.path.join(output_dir, "logreg_class_distribution.jpg")
metrics_heatmap_path = os.path.join(output_dir, "logreg_metrics_heatmap.jpg")
correlation_heatmap_path = os.path.join(output_dir, "logreg_correlation_heatmap.jpg")
roc_curve_path = os.path.join(output_dir, "logreg_roc_curve.jpg")

# ---------------------------------------
# 🔹 Confusion Matrix Plot (Neat Labels)
# ---------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)
plt.title("Confusion Matrix for Logistic Regression Model", fontsize=14)
plt.xticks(rotation=0)  # Keep labels readable
plt.yticks(rotation=0)
plt.savefig(conf_matrix_path, bbox_inches="tight")
plt.show()

# ---------------------------------------
# 🔹 Class Distribution Plot (Neat Labels)
# ---------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x=y_pred)
plt.xticks(ticks=[0, 1, 2, 3], labels=labels, fontsize=10, rotation=0)  # Keep labels readable
plt.title("Predicted Class Distribution", fontsize=14)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.savefig(class_dist_path, bbox_inches="tight")
plt.show()

# ---------------------------------------
# 🔹 Precision, Recall, F1-Score Heatmap (Neat Labels)
# ---------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(metrics_matrix, annot=True, cmap="coolwarm", xticklabels=metrics, yticklabels=labels)
plt.xlabel("Metrics", fontsize=12)
plt.ylabel("Classes", fontsize=12)
plt.title("Evaluation Metrics Heatmap (Precision, Recall, F1-Score)", fontsize=14)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.savefig(metrics_heatmap_path, bbox_inches="tight")
plt.show()

# ---------------------------------------
# 🔹 Correlation Heatmap (Neat Labels)
# ---------------------------------------
# Convert Predictions into DataFrame for Correlation Analysis
pred_probs_df = pd.DataFrame(y_pred_probs, columns=labels)

# Compute Correlation Matrix
correlation_matrix = pred_probs_df.corr()

# 🔷 **Plot Correlation Heatmap**
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", xticklabels=labels, yticklabels=labels)
plt.xlabel("Classes", fontsize=12)
plt.ylabel("Classes", fontsize=12)
plt.title("Correlation Heatmap of Predicted Probabilities (Logistic Regression)", fontsize=14)
plt.xticks(rotation=0)  # Keep labels readable
plt.yticks(rotation=0)
plt.savefig(correlation_heatmap_path, bbox_inches="tight")
plt.show()

# ---------------------------------------
# 🔹 ROC Curve for Multi-Class Classification (Neat Labels)
# ---------------------------------------
y_true_binary = label_binarize(test_labels, classes=[0, 1, 2, 3])  # Convert labels to one-hot encoding

plt.figure(figsize=(6, 4))

for i in range(len(labels)):
    fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{labels[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.50)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate (Recall)", fontsize=12)
plt.title("ROC Curve for Logistic Regression Model", fontsize=14)
plt.legend(loc="lower right", fontsize=10)  # Keep legend readable
plt.savefig(roc_curve_path, bbox_inches="tight")  # Save ROC Curve
plt.show()

print(f"\n✅ All plots saved in '{output_dir}'")
