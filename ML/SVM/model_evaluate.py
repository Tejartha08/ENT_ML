import os
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ✅ Define Output Directory
output_dir = "D:/ENT_DISEASE_DETECTION/svm"
os.makedirs(output_dir, exist_ok=True)

# ✅ Load Test Data & Model
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy")

svm_model = joblib.load("ent_disease_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# ✅ Normalize Test Data
test_features = scaler.transform(test_features)

# ✅ Predict on Test Set
y_pred = svm_model.predict(test_features)
y_pred_probs = svm_model.decision_function(test_features)  # Get decision function for probability-like scores

# ✅ Compute Accuracy
accuracy = accuracy_score(test_labels, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ✅ Classification Report
class_report = classification_report(test_labels, y_pred, output_dict=True)

# ✅ Confusion Matrix
conf_matrix = confusion_matrix(test_labels, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# ✅ Define Class Labels (Disease Names)
class_labels = ["Chronic Otitis", "Earwax Plug", "Myringosclerosis", "Normal Ear"]

# ==========================
# 🔥 HEATMAP 1: Confusion Matrix
# ==========================
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=class_labels, yticklabels=class_labels)
plt.xticks(rotation=0, ha="center")  
plt.yticks(rotation=0, va="center")  
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for SVM Model")

# ✅ Save Confusion Matrix Plot
conf_matrix_path = os.path.join(output_dir, "svm_confusion_matrix.jpg")
plt.savefig(conf_matrix_path, format='jpg', dpi=300)
plt.show()

# ==========================
# 🔥 HEATMAP 2: Precision, Recall, F1-score
# ==========================
# Extract precision, recall, f1-score values
metrics_matrix = np.array([[class_report[str(i)]['precision'], class_report[str(i)]['recall'], class_report[str(i)]['f1-score']]
                           for i in range(len(class_labels))])

# Define metric labels
metric_labels = ["Precision", "Recall", "F1-score"]

# Plot Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(metrics_matrix, annot=True, cmap="coolwarm", xticklabels=metric_labels, yticklabels=class_labels, fmt=".2f")
plt.xticks(rotation=0, ha="center")  
plt.yticks(rotation=0, va="center")  
plt.xlabel("Metrics")
plt.ylabel("Classes")
plt.title("SVM Model Performance Metrics")

# ✅ Save Performance Metrics Heatmap
performance_heatmap_path = os.path.join(output_dir, "svm_performance_heatmap.jpg")
plt.savefig(performance_heatmap_path, format='jpg', dpi=300)
plt.show()

# ==========================
# 🔥 HEATMAP 3: Correlation Heatmap
# ==========================
# Convert predicted probabilities to DataFrame
pred_probs_df = pd.DataFrame(y_pred_probs, columns=class_labels)

# Compute correlation matrix
correlation_matrix = pred_probs_df.corr()

# Plot Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", xticklabels=class_labels, yticklabels=class_labels)
plt.xticks(rotation=0, ha="center")  
plt.yticks(rotation=0, va="center")  
plt.title("Correlation Heatmap of Predicted Scores (SVM Model)")

# ✅ Save Correlation Heatmap
correlation_heatmap_path = os.path.join(output_dir, "svm_correlation_heatmap.jpg")
plt.savefig(correlation_heatmap_path, format='jpg', dpi=300)
plt.show()

# ==========================
# 🔥 CLASS DISTRIBUTION PLOT
# ==========================
plt.figure(figsize=(6, 4))
sns.countplot(x=y_pred)
plt.xticks(ticks=range(len(class_labels)), labels=class_labels, rotation=0, ha="center")  
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Predicted Class Distribution")

# ✅ Save Class Distribution Plot
class_dist_path = os.path.join(output_dir, "svm_class_distribution.jpg")
plt.savefig(class_dist_path, format='jpg', dpi=300)
plt.show()

# ==========================
# 🔥 ROC CURVE PLOT
# ==========================
# Binarize labels for multi-class ROC curve
y_test_binarized = label_binarize(test_labels, classes=[0, 1, 2, 3])

plt.figure(figsize=(6, 5))
for i in range(len(class_labels)):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_labels[i]} (AUC = {roc_auc:.2f})")

# Plot ROC Curve
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xticks(rotation=0, ha="center")  
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for SVM Model")
plt.legend(loc="lower right")

# ✅ Save ROC Curve Plot
roc_curve_path = os.path.join(output_dir, "svm_roc_curve.jpg")
plt.savefig(roc_curve_path, format='jpg', dpi=300)
plt.show()

print(f"\n✅ Plots saved successfully at:\n{conf_matrix_path}\n{performance_heatmap_path}\n{correlation_heatmap_path}\n{class_dist_path}\n{roc_curve_path}")
