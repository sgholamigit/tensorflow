#!/usr/bin/env python
""" Auto Encoder Example.
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import tflearn
import os
import sys
from IPython.display import display, Image
from scipy import ndimage
import matplotlib.image as mpimg
from PIL import Image
import h5py
import tensorflow
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d


image_size = 64  # Pixel width and height.

# function to load a file
def loadFile(filename):
  f = h5py.File(filename, 'r')
  dataset = f[f.keys()[0]][:]
  return dataset

# function to reformat to (-1, 256*256)
def reformat(dataset):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  return dataset

# function to extract encoded features
def build_encoded_features(dataset):
  #encoded_array_dataset = np.zeros((dataset.shape[0], 2))
  encoded_array_dataset = np.zeros((dataset.shape[0], 4, 4, 8))
  dataset = tflearn.data_utils.shuffle(dataset)[0]
  batch_size = 250
  for i in range(int(dataset.shape[0] / batch_size)):
      start = i*batch_size
      end = (i + 1) * batch_size
      if end > dataset.shape[0]:
          end = -1
      for encoded_features in encoding_model.predict(dataset[start:end]):
          encoded_array_dataset[start:end] = np.asarray(encoded_features)
  return encoded_array_dataset

# function to build encoded-decoded output data
def build_encode_decode(dataset):
  encode_decode_data = []
  dataset = tflearn.data_utils.shuffle(dataset)[0]
  batch_size = 250
  for i in range(int(dataset.shape[0] / batch_size)):
    start = i*batch_size
    end = (i + 1) * batch_size
    if end > dataset.shape[0]:
        end = -1
    for predicted_data in model.predict(dataset[start:end]):
        encode_decode_data.append(predicted_data)
  return encode_decode_data

files = os.listdir('./train')
files = ['./train/' + file for file in files]
for i, file in enumerate(files):
    f = h5py.File(file)
    X = f['X'][:]
train_dataset = X

files = os.listdir('./test')
files = ['./test/' + file for file in files]
for i, file in enumerate(files):
    f = h5py.File(file)
    X = f['X'][:]
test_dataset12 = X

files = os.listdir('./test2')
files = ['./test2/' + file for file in files]
for i, file in enumerate(files):
    f = h5py.File(file)
    X = f['X'][:]
test_dataset23 = X


encoder = input_data(shape=[None, image_size, image_size, 1])
encoder = conv_2d(encoder, 32, 3, activation='relu')
encoder = max_pool_2d(encoder, 2)
encoder = conv_2d(encoder, 32, 3, activation='relu')
encoder = max_pool_2d(encoder, 2)
encoder = conv_2d(encoder, 16, 3, activation='relu')
encoder = max_pool_2d(encoder, 2)
encoder = conv_2d(encoder, 8, 3, activation='relu')
encoder = max_pool_2d(encoder, 2)

decoder = conv_2d(encoder, 8, 3, activation='relu')
decoder = upsample_2d(decoder, 2)
decoder = conv_2d(decoder, 16, 3, activation='relu')
decoder = upsample_2d(decoder, 2)
decoder = conv_2d(decoder, 32, 3, activation='relu')
decoder = upsample_2d(decoder, 2)
decoder = conv_2d(decoder, 32, 3, activation='relu')
decoder = upsample_2d(decoder, 2)
decoder = conv_2d(decoder, 1, 3, activation='relu')







# Regression, with mean square error
net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
                         loss='mean_square', metric=None)

# Training the auto encoder
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(train_dataset, train_dataset, n_epoch=5, validation_set=0.1,
          batch_size=250)

# Encoding X[0] for test
print("\nTest encoding of X[0]:")
# New model, re-using the same session, for weights sharing
encoding_model = tflearn.DNN(encoder, session=model.session)

# Testing the image reconstruction on new data (test set)
print("\nVisualizing results after being encoded and decoded:")

#predict_train = encoding_model.predict([train_dataset12[0]])
encoded_array_test_dataset12 = build_encoded_features(test_dataset12)
print (encoded_array_test_dataset12.shape)



