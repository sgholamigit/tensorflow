#!/usr/bin/env python

import h5py
import sys
import numpy as np
import copy
import random
import argparse
import os
from deepNN import deepNN
import tflearn
from tflearn.layers.estimator import regression
from tflearn.data_utils import shuffle
import tensorflow as tf

def runDNN(X, Y, hx, hy, nch, test=False):
    dnn = deepNN(hx, hy, nch)
    dnn = regression(dnn, optimizer='adam',
                     loss=tflearn.mean_square,
                     learning_rate=0.001)

    model = tflearn.DNN(dnn)
    if test:
        model.load('auto_test.dnn')
        f = open('pred_vs_true2.dat', 'w')
        batch_size = 250
        for i in range(len(Y) / batch_size):
            start = i*batch_size
            end = (i + 1) * batch_size
            if end > len(Y):
                end = -1
            for predicted, true in zip(model.predict(X[start:end]), Y[start:end]):
                f.write('%f\t%f\n' % (predicted[0], true[0]))
        print('file is ready!')

    else:
	# model.load('dimer_test.dnn')
        model.fit(X, Y, batch_size=250, n_epoch=100, show_metric=False, run_id='dimer_test', validation_set=0.1)
        model.save('auto_test.dnn')

def loadInData(test=False):
    try:
        if not test:
            files = os.listdir('./trainExtracted')
            files = ['./trainExtracted/' + file for file in files]

        else:
            files = os.listdir('./test2Extracted')
            files = ['./test2Extracted/' + file for file in files]
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
    nch = None
    for i, file in enumerate(files):
        f = h5py.File(file)
        X, Y = f['X'], f['Y']
        new_hx, new_hy = f.attrs['image_size']
        new_nch = f.attrs['num_channels']
        if hx is None and hy is None and nch is None:
            hx, hy, nch = new_hx, new_hy, new_nch
        elif hx != new_hx or hy != new_hy or nch != new_nch:
            print 'Error: image sizes should be the same across all files.'
            exit(-1)
    return X, Y, hx, hy, nch

def parseArgs():
    parser = argparse.ArgumentParser(description='Regression using Tensorflow. Change deepNN to modify the neural network.')
    parser.add_argument('--train', help='Start training with data stored in "train" directory.', action='store_true', dest='train')
    parser.add_argument('--test', help='Start testing with data stored in "test" directory.', action='store_true', dest='test')
    parser.add_argument('--name', help='If testing selected, you must specify the name of the saved model.', dest='name', type=str)
    return parser.parse_args()

def main():
    parser = parseArgs()
    if parser.train:
        print 'Training selected.'
        X, Y, hx, hy, nch = loadInData()
        runDNN(X, Y, hx, hy, nch)
    elif parser.test:
        print 'Testing selected.'
        X, Y, hx, hy, nch = loadInData(test=True)
        runDNN(X, Y, hx, hy, nch, test=True)
    else:
        print 'Error: Training nor testing mode was selected. Use --help to see usage.'
        exit(-1)

if __name__ == '__main__':
    main()
