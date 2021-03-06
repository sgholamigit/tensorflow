from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from scipy import ndimage
from six.moves import cPickle as pickle
import tensorflow as tf


train_filename = 'train256classSimple10000'
test_filename = 'test256classSimple200'



# extracting the datasets
num_classes = 256
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)



#Merging the data

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
num_of_classes = 256

float_formatter = lambda x: "%.2f" % x

        
def Merge_folders(data_folders, size_per_class):
  dataset_names = []
  start_t = 0
  end_t = size_per_class
  required_size = size_per_class * num_of_classes
  trainDataset = np.ndarray((required_size, image_size, image_size), dtype=np.float32)
  labelsDataset = np.ndarray((required_size, 1), dtype=np.float32)
  for folder in data_folders:
    dataset_names.append(folder)

    print('Merging %s.' % folder)
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    image_index = 0
    for image in os.listdir(folder):
        if image_index < size_per_class:
            image_file = os.path.join(folder, image)
            image_data = (ndimage.imread(image_file).astype(int))
            dataset[image_index, :, :] = image_data
            image_index += 1
    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    trainDataset[start_t:end_t, :, :] = dataset
    x = float(folder.split("/")[1]) / 255.0
    #y = float_formatter(x)
    labelsDataset[start_t:end_t] = x
    start_t += size_per_class
    end_t += size_per_class

  
  return dataset_names, trainDataset, labelsDataset

train_datasets, trainDataset, labelsTrainDataset = Merge_folders(train_folders, 10000)
test_datasets, testDataset, labelsTestDataset = Merge_folders(test_folders, 200)

print(trainDataset)
print(labelsTrainDataset)

print(testDataset)
print(labelsTestDataset)

print('trainDataset.shape' , trainDataset.shape)
print('labelsTrainDataset.shape' , labelsTrainDataset.shape)

print('testDataset.shape' , testDataset.shape)
print('labelsTestDataset.shape' , labelsTestDataset.shape)


# randomize the data
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(trainDataset, labelsTrainDataset)
test_dataset, test_labels = randomize(testDataset, labelsTestDataset)
print('Randomizing is done')


# Reformat the data: image data as a cube (width by height by number 
# of channels) and labels as float 1-hot encodings.
image_size = 28
num_labels = 1
num_channels = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  #labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Reformatted training set', train_dataset.shape, train_labels.shape)
print('Reformatted test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def medianSqr(predictions, labels):
  print (labels.shape)
  print (predictions.shape)
  return (np.median(np.square(predictions - labels)))

def meanSqr(predictions, labels):
  print (labels.shape)
  print (predictions.shape)
  return (np.mean(np.square(predictions - labels)))

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

def drop_out(x, keep_prob):
  return tf.nn.dropout(x, keep_prob)


batch_size = 100


graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  W_conv1 = weight_variable([5, 5, 1, 20])
  b_conv1 = bias_variable([20])

  W_conv2 = weight_variable([5, 5, 20, 50])
  b_conv2 = bias_variable([50])

  W_fc1 = weight_variable([7 * 7 * 50, 500])
  b_fc1 = bias_variable([500])

  W_fc2 = weight_variable([500, num_labels])
  b_fc2 = bias_variable([num_labels])

  
  # Model.
  def model(data):
    h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    shape = h_pool2.get_shape().as_list()
    reshape = tf.reshape(h_pool2, [shape[0], shape[1] * shape[2] * shape[3]])
    h_fc1 = tf.nn.tanh(tf.matmul(reshape, W_fc1) + b_fc1)
    #keep_prob = tf.placeholder(tf.float32)
    #h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    return tf.matmul(h_fc1, W_fc2) + b_fc2
   
  
  # Training computation.
  logits = model(tf_train_dataset)
  #loss = tf.reduce_mean(
    #tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  loss = tf.reduce_mean(tf.square(logits - tf_train_labels))

    
  # Optimizer.
  optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  #train_prediction = tf.nn.softmax(logits)
  train_prediction = logits
  #test_prediction = tf.nn.softmax(model(tf_test_dataset))
  test_prediction = model(tf_test_dataset)


num_steps = 25601
step_plot = []
loss_plot = []
accuracy_plot = []

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()

  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      #acc = accuracy(predictions, batch_labels)
      #print('Minibatch accuracy: %.1f%%' % acc)
      step_plot.append(step)
      loss_plot.append(l)
      #accuracy_plot.append(acc)
  print('Test dataset median regression error: %.5f' % medianSqr(test_prediction.eval(), test_labels))
  print('Test dataset mean regression error: %.5f' % meanSqr(test_prediction.eval(), test_labels))
  plt.figure(facecolor='white')
  plt.plot(step_plot, loss_plot, 'rs--')
  plt.xlabel('Steps')
  plt.ylabel('Loss')
  plt.title('Loss plot')
  plt.grid(True)
  plt.show()
  #plt.figure(facecolor='white')
  #plt.plot(step_plot, accuracy_plot, '--bs')
  #plt.xlabel('Steps')
  #plt.ylabel('Accuracy')
  #plt.title('Accuracy plot')
  #plt.axis([0, 10000, 0, 110])
  #plt.grid(True)
  #plt.show()
  #print(accuracy_plot)
