import os
import sys
os.chdir(sys.path[0])

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
import skimage as ski
from matplotlib import pyplot as plt


class image_iterator:
    def __init__(self, collection):
        self.collection = collection
        self.current = 0
        self.high = len(collection)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.high:
            res = self.collection[self.current].flatten()
            self.current += 1
            return res
        raise StopIteration
    
    def reset(self):
        self.current = 0


train_images = image_iterator(ski.io.imread_collection("Data/linreg_img_class/prepped/train/*.jpg"))
train_y = [i % 2 for i in range(train_images.high)]

test_images = image_iterator(ski.io.imread_collection("Data/linreg_img_class/prepped/test/*.jpg"))
test_y = [i % 2 for i in range(test_images.high)]


model = LinearRegression()

params = None
set_params = True

for sample in zip(train_images, train_y):
    print("A!")
    model.fit([sample[0],], [sample[1],])
    if set_params:
        params = model.coef_
        set_params = False
        print(params)
    else:
        params = (params + model.coef_) / 2

print(params)

model.coef_ = params

predictions = []
for sample in test_images:
    print("B!")
    predictions.append(model.predict([sample]))

difference = np.abs(np.subtract(test_y, predictions)) # Absolute Errors
print(difference.mean()) # Mean Absolute Error (Which btw makes no sense for a binary classifier lmao)


# We achieved a majestic 0.5 MAE aka. It might as well be randomly guessing
# Surprise!!! Lin-Reg sucks at Image classification! :D

