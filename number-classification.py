import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define the path to the datasets
base_path = '/content/drive/My Drive/P2/dataset/'

def load_images_from_folder(folder):
    images = []
    labels = []
    if not os.path.exists(folder):
        print(f"Directory {folder} does not exist.")
        return np.array(images), np.array(labels)

    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img.flatten())
                labels.append(int(filename[0]))  # Extract label from filename (e.g., 0_01.png -> label is 0)
    if len(images) == 0:
        print(f"No images found in directory {folder}.")
    return np.array(images), np.array(labels)

# Load dataset a
train_images_a, train_labels_a = load_images_from_folder(base_path + 'a/Train')
test_images_a, test_labels_a = load_images_from_folder(base_path + 'a/Test')

# Check if data is loaded correctly
print(f"Loaded {len(train_images_a)} training images and {len(test_images_a)} test images for dataset a.")

def visualize_images(images, labels, title, image_shape, num_samples=40):
    plt.figure(figsize=(40, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        img = images[i].reshape(image_shape)
        plt.imshow(img, cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Visualize a few training images
if len(train_images_a) > 0:
    image_shape = (60, 60)
    visualize_images(train_images_a, train_labels_a, 'Training Images - Dataset a', image_shape)
if len(test_images_a) > 0:
    image_shape = (60, 60)
    visualize_images(test_images_a, test_labels_a, 'Test Images - Dataset a', image_shape, 20)

def calculate_prototypes(images, labels):
    classes = np.unique(labels)
    prototypes = {}
    for cls in classes:
        class_images = images[labels == cls]
        prototypes[cls] = np.mean(class_images, axis=0)
    return prototypes

# Calculate prototypes for dataset a
prototypes_a = calculate_prototypes(train_images_a, train_labels_a)

def visualize_prototypes(prototypes, title, image_shape):
    plt.figure(figsize=(10, 2))
    for i, (cls, prototype) in enumerate(prototypes.items()):
        plt.subplot(1, len(prototypes), i + 1)
        plt.imshow(prototype.reshape(image_shape), cmap='gray')
        plt.title(f'Class: {cls}')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Visualize prototypes for dataset a
visualize_prototypes(prototypes_a, 'Prototypes - Dataset a', image_shape)

def classify_mdc(test_images, prototypes):
    predictions = []
    for image in test_images:
        min_distance = float('inf')
        predicted_class = None
        for cls, prototype in prototypes.items():
            distance = np.linalg.norm(image - prototype)
            if distance < min_distance:
                min_distance = distance
                predicted_class = cls
        predictions.append(predicted_class)
    return np.array(predictions)

# Classify test samples for dataset a
predictions_a = classify_mdc(test_images_a, prototypes_a)

def visualize_classification_results(test_images, test_labels, predictions, title, image_shape, num_samples=20):
    plt.figure(figsize=(100, 4))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        img = test_images[i].reshape(image_shape)
        plt.imshow(img, cmap='gray')
        plt.title(f'True: {test_labels[i]}')
        plt.axis('off')

        plt.subplot(2, num_samples, num_samples + i + 1)
        img = test_images[i].reshape(image_shape)
        plt.imshow(img, cmap='gray')
        plt.title(f'Pred: {predictions[i]}')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Visualize classification results for dataset a
visualize_classification_results(test_images_a, test_labels_a, predictions_a, 'Classification Results - Dataset a', image_shape)

# Calculate error rate
error_a = np.mean(predictions_a != test_labels_a)
print(f'Error rate for dataset a: {error_a}')

# Load dataset b
train_images_b, train_labels_b = load_images_from_folder(base_path + 'b/Train')
test_images_b, test_labels_b = load_images_from_folder(base_path + 'b/Test')

# Check if data is loaded correctly
print(f"Loaded {len(train_images_b)} training images and {len(test_images_b)} test images for dataset b.")

# Visualize a few training images
if len(train_images_b) > 0:
    image_shape = (60, 60)
    visualize_images(train_images_b, train_labels_b, 'Training Images - Dataset b', image_shape)
if len(test_images_b) > 0:
    image_shape = (60, 60)
    visualize_images(test_images_b, test_labels_b, 'Test Images - Dataset b', image_shape, 20)

# Calculate prototypes for dataset b
prototypes_b = calculate_prototypes(train_images_b, train_labels_b)

# Visualize prototypes for dataset b
visualize_prototypes(prototypes_b, 'Prototypes - Dataset b', image_shape)

# Classify test samples for dataset b
predictions_b = classify_mdc(test_images_b, prototypes_b)

# Visualize classification results for dataset b
visualize_classification_results(test_images_b, test_labels_b, predictions_b, 'Classification Results - Dataset b', image_shape, 40)

# Calculate error rate
error_b = np.mean(predictions_b != test_labels_b)
print(f'Error rate for dataset b: {error_b}')

# Load dataset c
train_images_c, train_labels_c = load_images_from_folder(base_path + 'c/Train')
test_images_c, test_labels_c = load_images_from_folder(base_path + 'c/Test')

# Check if data is loaded correctly
print(f"Loaded {len(train_images_c)} training images and {len(test_images_c)} test images for dataset c.")

# Visualize a few training images
if len(train_images_c) > 0:
    image_shape = (60, 60)
    visualize_images(train_images_c, train_labels_c, 'Training Images - Dataset c', image_shape)
if len(test_images_c) > 0:
    image_shape = (60, 60)
    visualize_images(test_images_c, test_labels_c, 'Test Images - Dataset c', image_shape, 20)

# Calculate prototypes for dataset c
prototypes_c = calculate_prototypes(train_images_c, train_labels_c)

# Visualize prototypes for dataset c
visualize_prototypes(prototypes_c, 'Prototypes - Dataset c', image_shape)

# Classify test samples for dataset c
predictions_c = classify_mdc(test_images_c, prototypes_c)

# Visualize classification results for dataset c
visualize_classification_results(test_images=test_images_c, test_labels=test_labels_c, predictions=predictions_c, title='Classification Results - Dataset c', image_shape=image_shape, num_samples=100)

# Calculate error rate
error_c = np.mean(predictions_c != test_labels_c)
print(f'Error rate for dataset c: {error_c}')