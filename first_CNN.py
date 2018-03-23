# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 09:12:03 2018

@author: Jorge
"""

import numpy as np
import os
from tkinter.filedialog import askdirectory
import tifffile as tiff
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras. utils import np_utils

from matplotlib import pyplot as plt

from keras.models import load_model

#np.set_printoptions(threshold=np.nan)

#get the data from the appropiate folder
def get_data(data):
    path=askdirectory()
    os.chdir(path)
    for fname in os.listdir(path):
        image = tiff.imread(fname)
        data.append(image)
    label = [os.path.split(path)[1]]*len(data)
    return data, label

#Get the data and organize it in random arrays (label and data order is conserved)
data = []
data, label1 = get_data(data)
data, label2 = get_data(data)

data = np.array(data)
label1 = np.array(label1)
label2 = np.array(label2)

label = np.concatenate((label1, label2))

randomize = np.arange(data.shape[0])
np.random.shuffle(randomize)

data = data[randomize]
label = label[randomize]

#Definde diccionary to interpret the data
dicc = {"circle": 1, "square": 0}
label = np.where(label == "circle", 1, 0)

#Split sample into training and test
(X_train, X_test, Y_train, Y_test) = train_test_split(data, label, test_size= 0.25, random_state= 10)

#Preprocess the data until its suitable for the model
X_train = X_train.reshape(X_train.shape[0], 50, 50,1)
X_test = X_test.reshape(X_test.shape[0], 50, 50,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#This particular data is in 0/1, otherwise use:
"""
X_train /= 255
X_test /= 255
"""

"""
#definde model
model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(50,50,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) 

#compile model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Train the model
model.fit(X_train, Y_train, 
          batch_size=32, epochs=5, verbose=1)


path=askdirectory()
os.chdir(path)
model.save('first_CNN.h5')
"""
path=askdirectory()
os.chdir(path)
model = load_model('first_CNN.h5')

trial = X_test[0:20]
print (Y_test[0:20])


trial = trial.reshape(trial.shape[0],50,50,1)

prediction = model.predict(trial, batch_size=None, verbose=0, steps=None)
print ("Prediction: " + str(prediction))
