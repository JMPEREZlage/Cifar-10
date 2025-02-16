# Cifar-10
CNN Classifier for CIFAR-10 using TensorFlow 2

Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow 2 to classify images from the CIFAR-10 dataset. The model is trained and evaluated using Google Colab, with options to save trained models to Google Drive.

Features

Uses TensorFlow 2 and Keras for model building and training.

Implements a CNN architecture optimized for CIFAR-10 classification.

Includes model evaluation metrics and visualization.

Supports saving and loading models from Google Drive.

Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes include:

Airplane

Automobile

Bird

Cat

Deer

Dog

Frog

Horse

Ship

Truck

Installation

Install dependencies:

pip install tensorflow numpy matplotlib

Run the notebook in Google Colab and mount Google Drive if needed.

Usage

Load and preprocess the CIFAR-10 dataset.

Define and compile the CNN model.

Train the model and monitor accuracy.

Evaluate the model and visualize predictions.

Save and reload the trained model.

Model Architecture

The CNN model consists of multiple convolutional layers followed by max-pooling, dropout layers for regularization, and a fully connected dense output layer with softmax activation.

Results

The model achieves approximately 80% accuracy on the CIFAR-10 test set after training. Further improvements can be made using data augmentation and hyperparameter tuning.

Contributions

Feel free to fork this repository and contribute improvements.

License

This project is open-source and available under the MIT License.
