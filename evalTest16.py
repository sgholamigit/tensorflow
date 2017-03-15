from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from six.moves import cPickle as pickle
import tensorflow as tf
import argparse


train_filename = 'train256classSimple10000'
test_filename = 'test256classSimple1000'
pickle_file = 'br256classSimple1000Reg.pickle'

# Basic model parameters as external flags.
FLAGS = None

def run_training():
  try:
    pickle_file
    print ("pickle file exists!")
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      train_dataset = save['train_dataset']
      train_labels = save['train_labels']
      test_dataset = save['test_dataset']
      test_labels = save['test_labels']
      del save  # hint to help gc free up memory
      print('Training set', train_dataset.shape, train_labels.shape)
      print('Test set', test_dataset.shape, test_labels.shape)
  except NameError:
    print ("pickle file does not exist!")

    # extracting the datasets
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
      if len(data_folders) != FLAGS.number_classses:
        raise Exception(
          'Expected %d folders, one per class. Found %d instead.' % (
	    FLAGS.number_classses, len(data_folders)))
      print(data_folders)
      return data_folders
  
    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)

    #Merging the data
    float_formatter = lambda x: "%.2f" % x

    def Merge_folders(data_folders, size_per_class):
      dataset_names = []
      start_t = 0
      end_t = size_per_class
      required_size = size_per_class * FLAGS.number_classses
      train_dataset = np.ndarray((required_size, FLAGS.image_size, FLAGS.image_size), dtype=np.float32)
      labelsDataset = np.ndarray((required_size, 1), dtype=np.float32)
      for folder in data_folders:
        dataset_names.append(folder)

        print('Merging %s.' % folder)
        image_files = os.listdir(folder)
        dataset = np.ndarray(shape=(len(image_files), FLAGS.image_size, FLAGS.image_size),
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
        train_dataset[start_t:end_t, :, :] = dataset
        x = float(folder.split("/")[1]) / (FLAGS.pixel_depth)
        labelsDataset[start_t:end_t] = x
        start_t += size_per_class
        end_t += size_per_class

  
      return dataset_names, train_dataset, labelsDataset

    train_datasets, train_dataset, train_labels = Merge_folders(train_folders, 1000)
    test_datasets, test_dataset, test_labels = Merge_folders(test_folders, 100)

    print(train_dataset)
    print(train_labels)

    print(test_dataset)
    print(test_labels)

    print('train_dataset.shape' , train_dataset.shape)
    print('train_labels.shape' , train_labels.shape)

    print('test_dataset.shape' , test_dataset.shape)
    print('test_labels.shape' , test_labels.shape)

  # randomize the data
  def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
  train_dataset, train_labels = randomize(train_dataset, train_labels)
  test_dataset, test_labels = randomize(test_dataset, test_labels)
  print('Randomizing is done')


  # Reformat the data: image data as a cube (width by height by number 
  # of channels) and labels as float 1-hot encodings.
  num_labels = 1

  def reformat(dataset, labels):
    dataset = dataset.reshape(
      (-1, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels)).astype(np.float32)
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

  graph = tf.Graph()

  with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
      tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset)
  
    # Variables.
    W_conv1 = weight_variable([5, 5, 1, 6])
    b_conv1 = bias_variable([6])

    W_conv2 = weight_variable([5, 5, 6, 16])
    b_conv2 = bias_variable([16])

    W_fc1 = weight_variable([7 * 7 * 16, 120])
    b_fc1 = bias_variable([120])

    W_fc2 = weight_variable([120, 84])
    b_fc2 = bias_variable([84])

    W_fc3 = weight_variable([84, num_labels])
    b_fc3 = bias_variable([num_labels])

  
    # Model.
    def model(data):
      h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
      h_pool1 = max_pool_2x2(h_conv1)
      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
      h_pool2 = max_pool_2x2(h_conv2)
      shape = h_pool2.get_shape().as_list()
      reshape = tf.reshape(h_pool2, [shape[0], shape[1] * shape[2] * shape[3]])
      h_fc1 = tf.nn.tanh(tf.matmul(reshape, W_fc1) + b_fc1)
      h_fc2 = tf.nn.tanh(tf.matmul(h_fc1, W_fc2) + b_fc2) 
      return tf.matmul(h_fc2, W_fc3) + b_fc3
   
  
    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.square(logits - tf_train_labels))
    
    # Optimizer.
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = logits
    test_prediction = model(tf_test_dataset)

    step_plot = []
    loss_plot = []

    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    sess.run(init)

    print('Initialized')
    for step in range(FLAGS.max_steps):
      offset = (step * FLAGS.batch_size) % (train_labels.shape[0] - FLAGS.batch_size)
      batch_data = train_dataset[offset:(offset + FLAGS.batch_size), :, :, :]
      batch_labels = train_labels[offset:(offset + FLAGS.batch_size), :]
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
      _, l, predictions = sess.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)
      if (step % 50 == 0):
        print('Minibatch loss at step %d: %f' % (step, l))
        step_plot.append(step)
        loss_plot.append(l)
    print('Test dataset median regression error: %.5f' % medianSqr(test_prediction.eval(session=sess), test_labels))
    print('Test dataset medan regression error: %.5f' % meanSqr(test_prediction.eval(session=sess), test_labels))
    plt.figure(facecolor='white')
    plt.plot(step_plot, loss_plot, 'rs--')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss plot')
    plt.grid(True)
    plt.show()

def main(_):
  run_training()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.005,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=25601,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--number_classses',
      type=int,
      default=256,
      help='Number of classes in train and test data sets.'
  )
  parser.add_argument(
      '--image_size',
      type=int,
      default=28,
      help='Size of the image.'
  )
  parser.add_argument(
      '--pixel_depth',
      type=float,
      default=255.0,
      help='Number of levels per pixel.'
  )
  parser.add_argument(
      '--num_channels',
      type=int,
      default=1,
      help='1 for grayscale'
  )


  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


