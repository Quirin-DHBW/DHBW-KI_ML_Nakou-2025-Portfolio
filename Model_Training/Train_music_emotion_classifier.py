"""
VibeluX

Eat image, return emotion, use emotion to find songs :3
"""
################
## IMPORTS #####
################

if __name__ == "__main__":
    import pathlib
    import os
    import sys
    os.chdir(pathlib.Path(sys.path[0]).parent)

import tensorflow as tf
from tensorflow import keras as k

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#input()


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
        model.add(k.layers.MaxPooling2D((4, 4)))
        model.add(k.layers.Dropout(dropout))
    
    model.add(k.layers.Flatten())

    model.add(k.layers.Dense(128, activation='relu'))
    model.add(k.layers.Dropout(dropout))

    model.add(k.layers.Dense(128, activation='relu'))
    model.add(k.layers.Dropout(dropout))

    model.add(k.layers.Dense(64, activation='relu'))
    model.add(k.layers.Dropout(dropout))

    model.add(k.layers.Dense(7, activation='softmax'))

    model.compile(optimizer='adamw',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    return model


#######################
## Model training #####
#######################

train_dataset = create_dataset("Data/audio/Processed", batch_size=4, image_size=(1407, 1025))

conv_layers = [(28, (12, 12)), 
               (16, (12, 12)),
               (16, (12, 12))]

early_stopping = k.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)

model = create_model(conv_layers, dropout=0.05, input_size=(1407, 1025, 1))
history = model.fit(train_dataset, epochs=128, callbacks=[early_stopping])

model.save("music_emotion_classifier.h5")

