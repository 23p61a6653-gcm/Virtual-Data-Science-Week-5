# Week 5 – Deep Learning Application in Data Science

**Project:** Handwritten Digit Classification using CNN

## Dataset
MNIST Handwritten Digit Dataset, loaded through `keras.datasets.mnist`.

## Technologies
Python, TensorFlow/Keras, NumPy, Matplotlib.

## Model
Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling → Flatten → Dense(128) → Dropout(0.5) → Softmax(10).

## Training
Adam optimizer, sparse categorical cross-entropy, 10 epochs, batch size 128, 10% validation split.

## Run
Install TensorFlow, NumPy and Matplotlib, then run `python week5_mnist_cnn.py`. The MNIST dataset is downloaded automatically by Keras.

## Evaluation
The script prints test loss and test accuracy and plots training/validation accuracy.
