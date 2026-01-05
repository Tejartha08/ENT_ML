import os 
import cv2
import numpy as np
import scipy.signal
import skimage.feature as skf
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops

# ==========================
# STEP 1: Load Dataset, Resize, Normalize
# ==========================
def load_images_from_folder(folder):
    images = []
    labels = []
    categories = os.listdir(folder)  # Get class names

    for idx, category in enumerate(categories):
        img_files = glob(os.path.join(folder, category, "*.jpg"))  # Load only .jpg files

        for img_file in img_files:
            img = cv2.imread(img_file)  # Read the image
            img = cv2.resize(img, (224, 224))  # Resize to 224x224
            img = img / 255.0  # Normalize pixel values (0 to 1)

            images.append(img)
            labels.append(idx)  # Assign a numeric label to the category

    return np.array(images), np.array(labels)

# Load training and testing datasets
train_images, train_labels = load_images_from_folder("D:/ENT_DISEASE_DETECTION/Dataset/train")
test_images, test_labels = load_images_from_folder("D:/ENT_DISEASE_DETECTION/Dataset/test")

print("Train images shape:", train_images.shape)
print("Test images shape:", test_images.shape)

# ==========================
# STEP 2: Apply Wiener & Gaussian Filters
# ==========================
def apply_wiener_filter(image, kernel_size=5):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_image = scipy.signal.wiener(image_gray, kernel_size)
    return filtered_image

def apply_gaussian_filter(image, kernel_size=5):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.GaussianBlur(image_gray, (kernel_size, kernel_size), 0)
    return filtered_image

def apply_filters_to_dataset(images):
    wiener_filtered_images = []
    gaussian_filtered_images = []

    for img in tqdm(images, desc="Applying Filters"):
        img = (img * 255).astype(np.uint8)  # Convert back to 8-bit

        wiener_filtered = apply_wiener_filter(img)
        gaussian_filtered = apply_gaussian_filter(img)

        wiener_filtered_images.append(wiener_filtered)
        gaussian_filtered_images.append(gaussian_filtered)

    return np.array(wiener_filtered_images), np.array(gaussian_filtered_images)

train_wiener, train_gaussian = apply_filters_to_dataset(train_images)
test_wiener, test_gaussian = apply_filters_to_dataset(test_images)

print("Filtered Train Shape (Wiener):", train_wiener.shape)
print("Filtered Train Shape (Gaussian):", train_gaussian.shape)

# ==========================
# STEP 3: Feature Extraction (HOG, LBP, GLCM)
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

def extract_features_from_dataset(images):
    hog_features = []
    lbp_features = []
    glcm_features = []

    for img in tqdm(images, desc="Extracting Features"):
        img = (img * 255).astype(np.uint8)  # Fix: Convert back to uint8
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        hog_feat = extract_hog_features(img_gray)
        lbp_feat = extract_lbp_features(img_gray)
        glcm_feat = extract_glcm_features(img_gray)

        hog_features.append(hog_feat)
        lbp_features.append(lbp_feat)
        glcm_features.append(glcm_feat)

    return np.array(hog_features), np.array(lbp_features), np.array(glcm_features)

# Extract features
train_hog, train_lbp, train_glcm = extract_features_from_dataset(train_images)
test_hog, test_lbp, test_glcm = extract_features_from_dataset(test_images)

# Save extracted features
np.save("train_hog.npy", train_hog)
np.save("train_lbp.npy", train_lbp)
np.save("train_glcm.npy", train_glcm)
np.save("test_hog.npy", test_hog)
np.save("test_lbp.npy", test_lbp)
np.save("test_glcm.npy", test_glcm)

print("Train Features Shape - HOG:", train_hog.shape)
print("Train Features Shape - LBP:", train_lbp.shape)
print("Train Features Shape - GLCM:", train_glcm.shape)

# ==========================
# STEP 4: Feature Fusion and Saving Dataset
# ==========================

def fuse_features(hog_features, lbp_features, glcm_features):
    return np.hstack((hog_features, lbp_features, glcm_features))  # Horizontally stack features

# Fuse training and testing features
train_features = fuse_features(train_hog, train_lbp, train_glcm)
test_features = fuse_features(test_hog, test_lbp, test_glcm)

print("Fused Train Features Shape:", train_features.shape)  # (720, X)
print("Fused Test Features Shape:", test_features.shape)    # (160, X)

# Save final dataset for training
np.save("train_features.npy", train_features)
np.save("test_features.npy", test_features)
np.save("train_labels.npy", train_labels)
np.save("test_labels.npy", test_labels)

print("Features saved successfully!")

#---------------------------------------
# for test data 
#---------------------------------------
import os
import cv2
import numpy as np
import scipy.signal
import joblib
from glob import glob
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops

# Load Test Dataset
def load_images_from_folder(folder):
    images = []
    labels = []
    categories = os.listdir(folder)  # Get class names

    for idx, category in enumerate(categories):
        img_files = glob(os.path.join(folder, category, "*.jpg"))  # Load only .jpg files

        for img_file in img_files:
            img = cv2.imread(img_file)  # Read image
            if img is None:
                print(f"Skipping {img_file}, could not be loaded.")
                continue
            
            img = cv2.resize(img, (224, 224))  # Resize
            img = img / 255.0  # Normalize
            
            images.append(img)
            labels.append(idx)  # Assign category index as label

    return np.array(images), np.array(labels)

# Load Test Data
test_images, test_labels = load_images_from_folder("D:/ENT_DISEASE_DETECTION/Dataset/Test")
print("Test images loaded:", test_images.shape)

# ==========================
# STEP 1: Apply Filters
# ==========================
def apply_wiener_filter(image, kernel_size=5):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return scipy.signal.wiener(image_gray, kernel_size)

def apply_gaussian_filter(image, kernel_size=5):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(image_gray, (kernel_size, kernel_size), 0)

def apply_filters_to_dataset(images):
    wiener_filtered_images = []
    gaussian_filtered_images = []

    for img in tqdm(images, desc="Applying Filters"):
        img = (img * 255).astype(np.uint8)  # Convert to uint8

        wiener_filtered = apply_wiener_filter(img)
        gaussian_filtered = apply_gaussian_filter(img)

        wiener_filtered_images.append(wiener_filtered)
        gaussian_filtered_images.append(gaussian_filtered)

    return np.array(wiener_filtered_images), np.array(gaussian_filtered_images)

# Apply Filters
test_wiener, test_gaussian = apply_filters_to_dataset(test_images)

# ==========================
# STEP 2: Feature Extraction (HOG, LBP, GLCM)
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

def extract_features_from_dataset(images):
    hog_features = []
    lbp_features = []
    glcm_features = []

    for img in tqdm(images, desc="Extracting Features"):
        img = (img * 255).astype(np.uint8)  # Convert to uint8
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        hog_feat = extract_hog_features(img_gray)
        lbp_feat = extract_lbp_features(img_gray)
        glcm_feat = extract_glcm_features(img_gray)

        hog_features.append(hog_feat)
        lbp_features.append(lbp_feat)
        glcm_features.append(glcm_feat)

    return np.array(hog_features), np.array(lbp_features), np.array(glcm_features)

# Extract Features from Test Data
test_hog, test_lbp, test_glcm = extract_features_from_dataset(test_images)

# ==========================
# STEP 3: Feature Fusion
# ==========================
def fuse_features(hog_features, lbp_features, glcm_features):
    return np.hstack((hog_features, lbp_features, glcm_features))  # Horizontally stack features

# Fuse test features
test_features = fuse_features(test_hog, test_lbp, test_glcm)

print("Fused Test Features Shape:", test_features.shape)

# ==========================
# STEP 4: Save Processed Features
# ==========================
np.save("test_features.npy", test_features)
np.save("test_labels.npy", test_labels)

print("Test dataset processing complete. Features saved successfully!")
