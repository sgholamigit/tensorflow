from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


pickle_file = 'brightnessData.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

# reformat the data to flat matrix and one hot encodings
image_size = 28
num_labels = 256

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 256])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
#h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
#h_pool2 = max_pool_2x2(h_conv2)

#W_conv3 = weight_variable([5, 5, 64, 128])
#b_conv3 = bias_variable([128])
#h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
#h_pool3 = max_pool_2x2(h_conv3)

#W_conv4 = weight_variable([5, 5, 128, 256])
#b_conv4 = bias_variable([256])
#h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
#h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
#h_pool4 = max_pool_2x2(h_conv4)

#W_conv5 = weight_variable([5, 5, 256, 512])
#b_conv5 = bias_variable([512])
#h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
#h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
#h_pool5 = max_pool_2x2(h_conv5)

#W_conv6 = weight_variable([5, 5, 512, 1024])
#b_conv6 = bias_variable([1024])
#h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
#h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)
#h_pool6 = max_pool_2x2(h_conv6)

W_fc1 = weight_variable([28 * 28 * 64, 1024])
b_fc1 = bias_variable([1024])

#h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_conv2_flat = tf.reshape(h_conv2, [-1, 28*28*64])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 256])
b_fc2 = bias_variable([256])

#y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  offset = (i * 50) % (train_labels.shape[0] - 50)
  batch_xs = train_dataset[offset:(offset + 50), :]
  batch_ys = train_labels[offset:(offset + 50), :]
  
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch_xs, y_: batch_ys})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch_xs, y_: batch_ys})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test_dataset, y_: test_labels}))

