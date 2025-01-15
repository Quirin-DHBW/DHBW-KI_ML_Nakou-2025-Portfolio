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


def prepare_image(img):
    img = ski.transform.resize(img, (256, 256))

    flat_c = img.flatten(order="C")
    flat_f = img.flatten(order="F")

    interleaved = np.dstack((flat_c, flat_f, flat_c, flat_f)).flatten()

    return interleaved.reshape((512, 512, 3))

def prepare_collection(collection, path, name):
    n_images = len(collection)
    for img_ind in range(n_images):
        fname = path + "_" + str(img_ind) + "_" + name + ".jpg"

        img_uint_rgb = (prepare_image(collection[img_ind]) * 255).astype(np.uint8)

        ski.io.imsave(fname=fname, arr=img_uint_rgb)
        print(f"{img_ind + 1}/{n_images}", end="\r")
    print()

cat_train_images = ski.io.imread_collection("Data/linreg_img_class/training_set/training_set/cats/*.jpg")
dog_train_images = ski.io.imread_collection("Data/linreg_img_class/training_set/training_set/dogs/*.jpg")

cat_train_prepped = prepare_collection(cat_train_images, "Data/linreg_img_class/prepped/train/", "c-tr")
dog_train_prepped = prepare_collection(dog_train_images, "Data/linreg_img_class/prepped/train/", "d-tr")


cat_test_images = ski.io.imread_collection("Data/linreg_img_class/test_set/test_set/cats/*.jpg")
dog_test_images = ski.io.imread_collection("Data/linreg_img_class/test_set/test_set/dogs/*.jpg")

cat_test_prepped = prepare_collection(cat_test_images, "Data/linreg_img_class/prepped/test/", "c-te")
dog_test_prepped = prepare_collection(dog_test_images, "Data/linreg_img_class/prepped/test/", "d-te")

