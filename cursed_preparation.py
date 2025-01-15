"""
Hello, today I will be making something cursed:
Cat-Dog image classification using Linear regression...

Starting Naive Idea (that probably won't work):
- Force all images to share a format (eg. 500x500 px or something)
- Borrowing from convolution we're going to interleave the same image multiple times in different patterns
"""

import os
import sys
os.chdir(sys.path[0])

import numpy as np
import pandas as pd
import sklearn as sk
import skimage as ski
from matplotlib import pyplot as plt

import joblib

def prepare_image(img):
    img = ski.transform.resize(img, (500, 500))

    flat_c = img.flatten(order="C")
    flat_f = img.flatten(order="F")
    comb_a = flat_c + flat_f
    comb_b = flat_c - flat_f

    return np.dstack((flat_c, comb_a, flat_f, comb_b)).flatten()

def prepare_collection(collection):
    n_images = len(collection)
    res = []
    for img_ind in range(n_images):
        res.append(prepare_image(collection[img_ind]))
        print(f"{img_ind}/{n_images}", end="\r")
    print()
    return res

cat_train_images = ski.io.imread_collection("Data/linreg_img_class/training_set/training_set/cats/*.jpg")
dog_train_images = ski.io.imread_collection("Data/linreg_img_class/training_set/training_set/dogs/*.jpg")

cat_train_prepped = prepare_collection(cat_train_images)
dog_train_prepped = prepare_collection(dog_train_images)

joblib.dump((cat_train_prepped, dog_train_prepped), "CAT_DOG_DATASET_TRAIN.joblib")


cat_test_images = ski.io.imread_collection("Data/linreg_img_class/test_set/test_set/cats/*.jpg")
dog_test_images = ski.io.imread_collection("Data/linreg_img_class/test_set/test_set/dogs/*.jpg")

cat_test_prepped = prepare_collection(cat_test_images)
dog_test_prepped = prepare_collection(dog_test_images)

joblib.dump((cat_test_prepped, dog_test_prepped), "CAT_DOG_DATASET_TEST.joblib")

