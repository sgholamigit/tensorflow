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
  encoded_array_dataset = np.zeros((dataset.shape[0], 10))
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


train_filename = 'dimer12train.h5'
train_dataset = loadFile(train_filename)

test_filename12 = 'dimer12test.h5'
test_dataset12 = loadFile(test_filename12)

test_filename23 = 'dimer23test.h5'
test_dataset23 = loadFile(test_filename23)


image_size = 64  # Pixel width and height.


train_dataset = reformat(train_dataset)
#test_dataset12 = test_dataset12[0:1000, :]
test_dataset12 = reformat(test_dataset12)
#test_dataset23 = test_dataset23[0:1000, :]
test_dataset23 = reformat(test_dataset23)

# Building the encoder
encoder = tflearn.input_data(shape=[None, (image_size * image_size)])
encoder = tflearn.fully_connected(encoder, 10)

# Building the decoder
decoder = tflearn.fully_connected(encoder, (image_size*image_size))

# Regression, with mean square error
net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
                         loss='mean_square', metric=None)

# Training the auto encoder
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(train_dataset, train_dataset, n_epoch=100, validation_set=0.1,
          batch_size=250)

# Encoding X[0] for test
print("\nTest encoding of X[0]:")
# New model, re-using the same session, for weights sharing
encoding_model = tflearn.DNN(encoder, session=model.session)

# Testing the image reconstruction on new data (test set)
print("\nVisualizing results after being encoded and decoded:")
'''
# Applying encode and decode over test set
encode_decode12 = build_encode_decode(test_dataset12)
encode_decode23 = build_encode_decode(test_dataset23)

print("\nShowing the results:")
# Compare original images with their reconstructions
diff_sum12 = 0
diff_sum23 = 0

examples_to_show = 10
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(test_dataset12[i], (256, 256)), cmap='gray', interpolation='nearest')
    a[1][i].imshow(np.reshape(encode_decode12[i], (256, 256)), cmap='gray', interpolation='nearest')
f.show()
plt.draw()

for i in range(test_dataset12.shape[0]):
  diff_sum12 = diff_sum12 + (sum(abs(test_dataset12[i] - encode_decode12[i])))/65536

print ('Average test error per pixel 12 =', diff_sum12/test_dataset12.shape[0])


f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(test_dataset23[i], (256, 256)), cmap='gray', interpolation='nearest')
    a[1][i].imshow(np.reshape(encode_decode23[i], (256, 256)), cmap='gray', interpolation='nearest')
f.show()
plt.draw()

for i in range(test_dataset23.shape[0]):
  diff_sum23 = diff_sum23 + (sum(abs(test_dataset23[i] - encode_decode23[i])))/65536

print ('Average test error per pixel 23 =', diff_sum23/test_dataset23.shape[0])
'''

encoded_array_train = build_encoded_features(train_dataset)
encoded_array_test_dataset12 = build_encoded_features(test_dataset12)
encoded_array_test_dataset23 = build_encoded_features(test_dataset23)

train_mean = np.mean(abs(encoded_array_train), axis=0)
same_mean =  np.mean(abs(encoded_array_test_dataset12), axis=0)
nonsame_mean = np.mean(abs(encoded_array_test_dataset23), axis=0)

index = np.arange(0, 10)


print('encoded_array_train feature example', np.mean(abs(encoded_array_train), axis=0))
print('encoded_array_test_same feature example', np.mean(abs(encoded_array_test_dataset12), axis=0))
print('encoded_array_test_non_same feature example', np.mean(abs(encoded_array_test_dataset23), axis=0))

fig = plt.figure()
ax = fig.add_subplot(3, 1, 1)
ax.scatter(index, train_mean, color = 'r')
ax = fig.add_subplot(3, 1, 2)
ax.scatter(index,same_mean, color = 'g')
ax = fig.add_subplot(3, 1, 3)
ax.scatter(index, nonsame_mean, color = 'b')

'''
x = encoded_array_train[:,0]
y = encoded_array_train[:,1]
print (encoded_array_train.shape)
fig = plt.figure()
ax = fig.add_subplot(3, 1, 1)
ax.scatter(x, y, color = 'g')
plt.title('ising train dataset extracted features')


x = encoded_array_test_dataset12[:,0]
y = encoded_array_test_dataset12[:,1]
ax = fig.add_subplot(3, 1, 2)
ax.scatter(x, y, color = 'r')
plt.title('ising test dataset extracted features(same)')


x = encoded_array_test_dataset23[:,0]
y = encoded_array_test_dataset23[:,1]
ax = fig.add_subplot(3, 1, 3)
ax.scatter(x, y, color = 'b')
plt.title('ising test dataset extracted features(non same)')


'''
plt.waitforbuttonpress()


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
        #X = reformat(X)
        runANN(X, hx, hy)
    elif parser.test:
        print 'Testing selected.'
        X, hx, hy = loadInData(test=True)
        #X = reformat(X)
        runANN(X, hx, hy, test=True)
    else:
        print 'Error: Training nor testing mode was selected. Use --help to see usage.'
        exit(-1)

if __name__ == '__main__':
    main()

