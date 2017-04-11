#!/usr/bin/env python
""" Auto Encoder Example.
"""
import h5py
import sys
import numpy as np
import copy
import random
import argparse
import os
from autoNN2 import autoNN2
import tflearn
from tflearn.layers.estimator import regression
from tflearn.data_utils import shuffle
import tensorflow as tf
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import display, Image
from scipy import ndimage
import matplotlib.image as mpimg
from PIL import Image


image_size = 64  # Pixel width and height.




def initH5Py(filename, n_images, extracted_image_size, num_filters):
    os.chdir('/home/sgholami/distance_data64/1_2/')
    if not os.path.exists('test2Extracted'):
        os.makedirs('test2Extracted')
    os.chdir('test2Extracted')

    h5file = h5py.File(filename + '.h5', 'w')
    h5file.create_dataset('X', (n_images, extracted_image_size, extracted_image_size, num_filters))
    h5file.create_dataset('Y', (n_images, 1))     
    h5file.attrs['image_size'] = (int(extracted_image_size), int(extracted_image_size))
    h5file.attrs['num_channels'] = (int(num_filters))
    return h5file


def writeH5File(images, h5file, encoding_model, n_images):
    print 'started writing to the file'
    images = shuffle(images)
    labels = np.ones(n_images)
    labels = labels.reshape((len(labels), 1))
    #energies = np.array(energies, dtype=np.float32)
    batch_size = 250
    for i in range(int(n_images / batch_size)):
      start = i*batch_size
      end = (i + 1) * batch_size
      if end > n_images:
          end = -1
      images = np.array(images).reshape((n_images, image_size, image_size, 1)).astype(np.float32)
      h5file['X'][start:end] = np.array(encoding_model.predict(images[start:end]), dtype=np.float32)
      h5file['Y'][start:end] = np.array(labels[start:end], dtype=np.float32)


def testH5File(h5file):
    print("train dataset Keys: %s" % h5file.keys())
    print (h5file.keys()[0])
    data = h5file[h5file.keys()[0]][:]
    labels = h5file[h5file.keys()[1]][:]
    print (data.shape)
    print (labels.shape)



def runANN(X, hx, hy):    
    encoder, ann, extracted_image_size, num_filters = autoNN2(hx, hy)
    net = regression(ann, optimizer='adam',
                     loss=tflearn.mean_square,
                     learning_rate=0.001,
                     metric=None)
    model = tflearn.DNN(net)    
    model.fit(X, X, batch_size=256, n_epoch=50, validation_set=0.01, show_metric=False, run_id='auto_test')
    encoding_model = tflearn.DNN(encoder, session=model.session)
    return encoding_model, extracted_image_size,num_filters



def loadInData():
    try:
        files = os.listdir('./test2')
        files = ['./test2/' + file for file in files]
    except:
        print 'Error: You must have your h5 files in "train" directory.'
        exit(-1)

    if len(files) == 0:
        print 'Error: no h5 files found.'
        exit(-1)

    hx = None
    hy = None
    for i, file in enumerate(files):
        f = h5py.File(file)
        X = f['X'][:]
        new_hx, new_hy = f.attrs['image_size']
        if hx is None and hy is None:
            hx, hy = new_hx, new_hy
        elif hx != new_hx or hy != new_hy:
            print 'Error: image sizes should be the same across all files.'
            exit(-1)
    return X, hx, hy



def main():
    X, hx, hy = loadInData()
    n_images = X.shape[0]
    encoding_model, extracted_image_size, num_filters = runANN(X, hx, hy)
    h5file = initH5Py('test2Features', n_images, extracted_image_size, num_filters)
    writeH5File(X, h5file, encoding_model, n_images)
    testH5File(h5file)


if __name__ == '__main__':
    main()

