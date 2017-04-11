#!/usr/bin/env python

import h5py
import sys
import numpy as np
import copy
import random
import argparse
import os
from autoNN import autoNN
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


image_size = 256

def reformat(dataset):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  return dataset


def runANN(X, hx, hy, test=False):
    
    num_pixels = hx * hy
    # Building the encoder
    encoder = input_data(shape=[None, num_pixels])
    #encoder = fully_connected(encoder, 2048)
    encoder = fully_connected(encoder, 10)

    # Building the decoder
    #decoder = fully_connected(encoder, 2048)
    decoder = fully_connected(encoder, num_pixels)

    net = regression(decoder, optimizer='adam',
                     loss=tflearn.mean_square,
                     learning_rate=0.001,
                     metric=None)

    model = tflearn.DNN(net)
    
    if test:
        model.load('auto_test.dnn')
        X = tflearn.data_utils.shuffle(X)[0]
        batch_size = 200
        encode_decode_data = np.zeros((X.shape[0], 65536))
        diff_sum = 0
        for i in range(X.shape[0] / batch_size):
            start = i*batch_size
            end = (i + 1) * batch_size
            if end > X.shape[0]:
                end = -1
            encode_decode_data[start:end] = np.asarray(model.predict(X[start:end]))

        for i in range(encode_decode_data.shape[0]):
            diff_sum = diff_sum + (sum(abs(X[i] - encode_decode_data[i])))/65536
        print ('Average test error per pixel =', diff_sum/X.shape[0])

        examples_to_show = 10
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            a[0][i].imshow(np.reshape(X[i], (256, 256)), cmap='gray', interpolation='nearest')
            a[1][i].imshow(np.reshape(encode_decode_data[i], (256, 256)), cmap='gray', interpolation='nearest')
        f.show()
        plt.draw()
        plt.waitforbuttonpress()
    else:
        model.fit(X, X, batch_size=200, n_epoch=50, validation_set=0.01, show_metric=False, run_id='auto_test')
        model.save('auto_test.dnn')
        #encoding_model = tflearn.DNN(encoder, session=model.session)
        #encoding_model.save('auto_test2.tfl')
        #return encoding_model
         

def loadInData(test=False):
    try:
        if not test:
            files = os.listdir('./train')
            files = ['./train/' + file for file in files]

        else:
            files = os.listdir('./test')
            files = ['./test/' + file for file in files]
    except:
        if test:
            print 'Error: You must have your h5 files in "test" directory.'
        else:
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

def parseArgs():
    parser = argparse.ArgumentParser(description='Autoencoder net using Tensorflow. Change autoNN to modify the neural network.')
    parser.add_argument('--train', help='Start training with data stored in "train" directory.', action='store_true', dest='train')
    parser.add_argument('--test', help='Start testing with data stored in "test" directory.', action='store_true', dest='test')
    parser.add_argument('--name', help='If testing selected, you must specify the name of the saved model.', dest='name', type=str)
    return parser.parse_args()

def main():
    parser = parseArgs()
    if parser.train:
        print 'Training selected.'
        X, hx, hy = loadInData()
        X = reformat(X)
        runANN(X, hx, hy)
    elif parser.test:
        print 'Testing selected.'
        X, hx, hy = loadInData(test=True)
        X = reformat(X)
        runANN(X, hx, hy, test=True)
    else:
        print 'Error: Training nor testing mode was selected. Use --help to see usage.'
        exit(-1)

if __name__ == '__main__':
    main()

