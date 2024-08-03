# Image Classification using Minimum Distance Classifier

This project demonstrates an image classification pipeline using a Minimum Distance Classifier (MDC). The project processes images from three datasets (a, b, and c), visualizes the data, calculates class prototypes, and evaluates classification performance.

## Project Overview

The project includes the following steps:
1. Load images from three datasets (a, b, and c).
2. Visualize a few sample images from the training and test sets.
3. Calculate prototypes for each class in the datasets.
4. Visualize the calculated prototypes.
5. Classify test images using the Minimum Distance Classifier.
6. Visualize the classification results.
7. Calculate and display the error rate for each dataset.

## Installation

To run this project, you need to have Python installed along with the following libraries:
- numpy
- matplotlib
- opencv-python
- google-colab

You can install the required libraries using the following command:
```bash
pip install numpy matplotlib opencv-python
```

## Usage

1. **Mount Google Drive**: The script mounts Google Drive to access datasets stored in it.
   - Ensure your datasets are stored in a directory named `P2/dataset` in your Google Drive.

2. **Load and Process Data**: The script loads images from the specified directories, processes them, and visualizes the samples and prototypes.

3. **Classification**: It classifies test images using prototypes calculated from the training images and visualizes the classification results.

4. **Evaluation**: It calculates and prints the error rate for each dataset.

### Directory Structure

The expected directory structure in Google Drive:
```
/My Drive/P2/dataset/
  ├── a/
  │   ├── Train/
  │   └── Test/
  ├── b/
  │   ├── Train/
  │   └── Test/
  └── c/
      ├── Train/
      └── Test/
```

### Example Code

The main parts of the code are:
- **Loading Images**:
  ```python
  def load_images_from_folder(folder):
      # code to load images
  ```
- **Visualizing Images**:
  ```python
  def visualize_images(images, labels, title, image_shape, num_samples=40):
      # code to visualize images
  ```
- **Calculating Prototypes**:
  ```python
  def calculate_prototypes(images, labels):
      # code to calculate prototypes
  ```
- **Classifying with MDC**:
  ```python
  def classify_mdc(test_images, prototypes):
      # code to classify images
  ```
- **Visualizing Classification Results**:
  ```python
  def visualize_classification_results(test_images, test_labels, predictions, title, image_shape, num_samples=20):
      # code to visualize classification results
  ```

### Running the Script

Run the script in Google Colab to execute the image classification pipeline. Ensure your datasets are correctly placed in your Google Drive, and then run the script to see the results.

## Results

The script will output the following:
- Number of training and test images loaded for each dataset.
- Visualization of a few sample images from the training and test sets.
- Prototypes calculated for each class.
- Visualization of classification results.
- Error rate for each dataset.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
