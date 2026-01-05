import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load extracted features and labels
train_features = np.load("train_features.npy")
test_features = np.load("test_features.npy")
train_labels = np.load("train_labels.npy")
test_labels = np.load("test_labels.npy")

print("Train Features Shape:", train_features.shape)
print("Test Features Shape:", test_features.shape)

# Normalize features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Train Logistic Regression Model
log_reg_model = LogisticRegression(max_iter=1000, random_state=42, multi_class="ovr")  # One-vs-Rest classification
log_reg_model.fit(train_features, train_labels)

# Predict on test set
y_pred = log_reg_model.predict(test_features)

# Evaluate the model
accuracy = accuracy_score(test_labels, y_pred)
print(f"\nLogistic Regression Model Accuracy: {accuracy * 100:.2f}%")

# Display Classification Report
print("\nClassification Report:\n", classification_report(test_labels, y_pred))

# Compute Confusion Matrix
conf_matrix = confusion_matrix(test_labels, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Save the trained model and scaler
joblib.dump(log_reg_model, "ent_disease_logreg_model.pkl")
joblib.dump(scaler, "logreg_scaler.pkl")

print("\n✅ Logistic Regression model and scaler saved successfully!")

# -----------------------
# 📊 Visualization Section
# -----------------------

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Chronic Otitis", "Myringosclerosis", "Normal Ear", "Ear Wax"],
            yticklabels=["Chronic Otitis", "Myringosclerosis", "Normal Ear", "Ear Wax"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for Logistic Regression Model")
plt.show()

# Plot Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y_pred)
plt.xticks(ticks=[0, 1, 2, 3], labels=["Chronic Otitis", "Myringosclerosis", "Normal Ear", "Ear Wax"])
plt.title("Predicted Class Distribution")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.show()
