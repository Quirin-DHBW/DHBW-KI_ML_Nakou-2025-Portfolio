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
from sklearn.model_selection import train_test_split

seed = 42

df = pd.read_csv("Data/IRIS.csv")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input((4,)))
model.add(tf.keras.layers.Dense(2))
model.add(tf.keras.layers.Dense(2))
model.add(tf.keras.layers.Dense(3, activation="softmax"))

model.compile(optimizer=tf.keras.optimizers.AdamW(), loss=tf.keras.losses.CategoricalCrossentropy())
model.summary()

df_X = df[df.columns[0:4]]
df_Y = df[df.columns[4]]

codes, uniques = pd.factorize(df_Y)

y_onehot = tf.keras.utils.to_categorical(codes, num_classes=3)
print(uniques)

x_train, x_test, y_train, y_test = train_test_split(df_X, y_onehot, test_size=0.1, random_state=seed)

model.fit(x=x_train, y=y_train, epochs=1000)

final_score = model.evaluate(x_test, y_test)

print("\n\nWow, a loss of:", final_score)

