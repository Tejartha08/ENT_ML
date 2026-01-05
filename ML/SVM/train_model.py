import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ==========================
# STEP 5: Model Selection & Training
# ==========================

# Load extracted features and labels
train_features = np.load("train_features.npy")
test_features = np.load("test_features.npy")
train_labels = np.load("train_labels.npy")
test_labels = np.load("test_labels.npy")

print("Train Features Shape:", train_features.shape)
print("Test Features Shape:", test_features.shape)

# Normalize features (Scaling for better performance)
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Train SVM Model
svm_model = SVC(kernel="linear", C=1.0, random_state=42)
svm_model.fit(train_features, train_labels)

# Predict on test set
y_pred = svm_model.predict(test_features)

# Evaluate the model
accuracy = accuracy_score(test_labels, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Display Classification Report
print("\nClassification Report:\n", classification_report(test_labels, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(test_labels, y_pred))

# Save the trained model and scaler
joblib.dump(svm_model, "ent_disease_svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully as 'ent_disease_svm_model.pkl' and 'scaler.pkl'!")


# Train the SVM model
# Print accuracy, classification report, and confusion matrix
# Save the trained model as ent_disease_svm_model.pkl