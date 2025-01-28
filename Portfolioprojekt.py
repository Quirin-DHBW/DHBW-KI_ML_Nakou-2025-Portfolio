"""
Portfolioprojekt :3

Things to analyse? Uh... no clue man.
"""

import os
import sys
os.chdir(sys.path[0])

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras as k
import keras_tuner as kt

from sklearn.model_selection import train_test_split


seed = 42

df = pd.read_csv("Data/IRIS.csv")

def make_model(hp):
    model = k.Sequential()

    model.add(k.layers.Input((4,)))
    hp_layers = hp.Int('layers', min_value=1, max_value=4, step=1)
    hp_cells = hp.Int('cells', min_value=1, max_value=12, step=1)

    for _ in range(hp_layers):
        model.add(k.layers.Dense(hp_cells))
    model.add(k.layers.Dense(3, activation="softmax"))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=k.optimizers.AdamW(learning_rate=hp_learning_rate),
                  loss=k.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    #model.summary()
    return model

df_X = df[df.columns[0:4]]
df_Y = df[df.columns[4]]

codes, uniques = pd.factorize(df_Y)

y_onehot = k.utils.to_categorical(codes, num_classes=3)
print(uniques)

x_train, x_test, y_train, y_test = train_test_split(df_X, y_onehot, test_size=0.1, random_state=seed)


tuner = kt.RandomSearch(make_model,
                        "val_accuracy")

stop_early = k.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(x_train, y_train, epochs=100, validation_split=0.1, callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best Layer Count:", best_hps.get("layers"))
print("Best Cell Count:", best_hps.get("cells"))
print("Best LR:", best_hps.get("learning_rate"))

model = tuner.hypermodel.build(best_hps)
model.fit(x_train, y_train, epochs=100)

final_score = model.evaluate(x_test, y_test)

print("\n\nLoss:", final_score[0], "\nAccuracy:", final_score[1])

