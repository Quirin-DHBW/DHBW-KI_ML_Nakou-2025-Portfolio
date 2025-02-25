"""
VibeluX

Eat image, return emotion, use emotion to find songs :3
"""

import os
import sys
os.chdir(sys.path[0])

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras as k

import matplotlib.pyplot as plt

seed = 42
# Define constants
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 35
DATA_DIR = "Data/archive"

# Function to create dataset
def create_dataset(directory, batch_size=BATCH_SIZE):
    return k.preprocessing.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        color_mode="grayscale",  # Since images are greyscale
        label_mode="int"  # Labels are inferred from directory names
    )

# Load datasets
train_dataset = create_dataset(os.path.join(DATA_DIR, "train"))
test_dataset = create_dataset(os.path.join(DATA_DIR, "test"))

# Function to create CNN model with customizable layers
def create_model(conv_layers, dropout=0.25):
    model = k.Sequential()
    model.add(k.Input((48, 48, 1)))
    model.add(k.layers.Rescaling(1./255))  # Normalize pixel values
    
    for filters, kernel_size in conv_layers:
        model.add(k.layers.Conv2D(filters, kernel_size, activation='relu', padding='same'))
        model.add(k.layers.MaxPooling2D((2, 2)))
        model.add(k.layers.Dropout(dropout))
    
    model.add(k.layers.Flatten())

    model.add(k.layers.Dense(128, activation='leaky_relu'))
    model.add(k.layers.Dropout(dropout))

    model.add(k.layers.Dense(64, activation='leaky_relu'))
    model.add(k.layers.Dropout(dropout))

    model.add(k.layers.Dense(7, activation='softmax'))

    model.compile(optimizer='adamw',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Define convolutional layers
conv_layers = [(64, (5, 5)), 
               (32, (4, 4)), 
               (32, (3, 3))]

# Define EarlyStopping callback
early_stopping = k.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Create and train the model
model = create_model(conv_layers)
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, callbacks=[early_stopping])

# Function to plot training results
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over epochs')
    
    plt.show()

# Plot training results
plot_history(history)

# Save the trained model
model.save("emotion_classifier.h5")

