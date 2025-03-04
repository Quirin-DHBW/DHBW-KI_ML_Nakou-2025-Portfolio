"""
VibeluX

Eat image, return emotion, use emotion to find songs :3
"""
################
## IMPORTS #####
################

import os
import sys
os.chdir(sys.path[0])

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras as k

import matplotlib.pyplot as plt


#############################
## FUNCTION DEFINITIONS #####
#############################

def create_dataset(directory, batch_size=64, image_size=(48, 48)):
    return k.preprocessing.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        color_mode="grayscale",
        label_mode="int"
    )


def create_model(conv_layers, dropout=0.25, input_size=(48, 48, 1)):
    """
    Create a Convolutional network :)
    """
    model = k.Sequential()
    model.add(k.Input(input_size))
    # Normalize pixels
    model.add(k.layers.Rescaling(1./255))
    
    for filters, kernel_size in conv_layers:
        model.add(k.layers.Conv2D(filters, kernel_size, activation='relu', padding='same'))
        model.add(k.layers.MaxPooling2D((2, 2), padding="same"))
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


#######################
## Model training #####
#######################

train_dataset = create_dataset("Data/archive/train")
test_dataset = create_dataset("Data/archive/test")

conv_layers = [(64, (5, 5)), 
               (32, (4, 4)), 
               (32, (3, 3))]

early_stopping = k.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model = create_model(conv_layers)
history = model.fit(train_dataset, epochs=100, validation_data=test_dataset, callbacks=[early_stopping])


def plot_history(history):
    # Plot accuracy and loss using model training history.
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


plot_history(history)

model.save("face_emotion_classifier.h5")

