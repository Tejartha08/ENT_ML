import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops

# ==========================
# Load the trained model and scaler
# ==========================
svm_model = joblib.load("ent_disease_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define class names
class_names = ["Chronic Otitis Media", "Earwax Plug", "Myringosclerosis", "Normal"]

# ==========================
# Define Feature Extraction Functions (HOG, LBP, GLCM)
# ==========================
def extract_hog_features(image):
    features, _ = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), 
                      block_norm='L2-Hys', visualize=True)
    return features

def extract_lbp_features(image, radius=3, n_points=24):
    lbp = local_binary_pattern(image, P=n_points, R=radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
    return hist

def extract_glcm_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, correlation, energy, homogeneity]

# ==========================
# Load and preprocess test images
# ==========================
test_folder = "D:/ENT_DISEASE_DETECTION/Dataset/test"
image_paths = []
test_features = []

for category in os.listdir(test_folder):
    category_path = os.path.join(test_folder, category)
    if os.path.isdir(category_path):  # Ensure it's a directory
        for image_file in os.listdir(category_path):
            img_path = os.path.join(category_path, image_file)
            image_paths.append(img_path)

            # Load the image
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Extract features
            hog_feat = extract_hog_features(img_gray)
            lbp_feat = extract_lbp_features(img_gray)
            glcm_feat = extract_glcm_features(img_gray)

            # Fuse features
            fused_features = np.hstack((hog_feat, lbp_feat, glcm_feat))
            test_features.append(fused_features)

# Convert to numpy array and normalize
test_features = np.array(test_features)
test_features = scaler.transform(test_features)

# ==========================
# Predict using the trained SVM model
# ==========================
predictions = svm_model.predict(test_features)
predicted_labels = [class_names[pred] for pred in predictions]

# ==========================
# Visualize Predictions
# ==========================
plt.figure(figsize=(12, 8))

for i in range(min(10, len(image_paths))):  # Display up to 10 images
    img = cv2.imread(image_paths[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_labels[i]}", fontsize=10)
    plt.axis("off")

plt.tight_layout()
plt.show()
