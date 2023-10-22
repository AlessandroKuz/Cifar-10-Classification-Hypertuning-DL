# CIFAR-10 Classification and Hyperparameter Tuning with Keras Tuner

This repository contains a comprehensive exploration of deep learning techniques, focusing on image classification using the CIFAR-10 dataset. The project is divided into several key components:

## 1. Libraries Import

Import essential libraries, including TensorFlow, Keras Tuner, and visualization tools.

## 2. Dataset Import and Preprocessing

Load the CIFAR-10 dataset and preprocess the images, scaling them to the range [0, 1].

## 3. Model Building and Hyperparameter Tuning

### 3.1 Hyperparameter Tuning

Utilize Keras Tuner to search for optimal hyperparameters for the convolutional neural network (CNN) model. The hyperparameters include filter sizes, kernel sizes, regularization strengths, and learning rates.

### 3.2 Model Training and Evaluation

Train the CNN model using the optimal hyperparameters and evaluate its performance on the test set. Visualize accuracy and loss curves for model evaluation.

## 4. Best Model Recreation and Evaluation

Recreate the best-performing CNN model, optimize data handling with TensorFlow datasets, and evaluate its performance. Visualize the confusion matrix and showcase both misclassified and correctly classified images.

## 5. Plotting Results and Confusion Matrix

Explore the project's results through detailed visualizations. Plot accuracy and loss curves to understand the model's training process. Utilize a confusion matrix to gain insights into the model's classification performance on the test data.

## Installation

To run the project, make sure to install the necessary modules listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

For detailed implementation and usage, refer to the Jupyter Notebook files in the repository.

Enjoy exploring the world of deep learning and image classification with CIFAR-10!