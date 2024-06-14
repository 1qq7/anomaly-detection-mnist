# Anomaly Detection in Handwritten Digit Recognition

This repository contains a PyTorch implementation exploring effective methods for detecting anomalous inputs in classification models using the MNIST dataset. While the primary focus is not solving handwritten digit recognition, the MNIST dataset provides a simple and illustrative context for our experiments.


## Introduction

Classification models often struggle with out-of-distribution (OOD) inputs, making incorrect and overconfident predictions. This project aims to explore the use of autoencoders integrated into a classification model to detect anomalous inputs based on reconstruction errors.

## Model Architecture

The model consists of three main components:
1. **Encoder**: A convolutional neural network (CNN) that extracts features from input images.
2. **Classifier**: A fully connected layer that predicts digit classes based on extracted features.
3. **Autoencoder**: A separate autoencoder network that reconstructs the features to compute reconstruction errors.

The goal is to train a model that can classify digits while also identifying inputs that do not belong to any of the digit classes (0-9) by analyzing reconstruction errors.

## Training
Train the model using the MNIST dataset:
```
python train.py
```
This script will train the model and save the trained weights in the model directory.

## Reconstruction Error Calculation
Calculate the reconstruction errors for each class and save the results:
```
python compute_class_reconstruction_errors.py
```
This script will create a file `reconstruction_errors.txt` containing the average reconstruction errors for each class.

## Testing
Test the model with the MNIST test set and detect anomalies:
```
python test.py
```
This script will load the model and reconstruction_errors.txt, then detect anomalies in the test set and save anomaly images in the anomalies directory.

## Generate Letter Images
Generate synthetic images of letters to test the anomaly detection:
```
python generate_letters.py
```
This script will generate letter images and save them in the letters directory.

## Test Letters
Test the model with letters and detect anomalies:
```
python test_letters.py
```

## Working with ChatGPT
[prompt](prompt)
