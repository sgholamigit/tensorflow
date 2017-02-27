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
from funcAttach import maybe_extract


#pickle_file = 'trainset7Simple10000.pickle'
#pickle_file = 'brSimpleclass30.pickle'
#pickle_file = 'train10step7Domain10000.pickle'
#pickle_file = 'train5stepSimple10000.pickle'
pickle_file = 'brSimnple26class.pickle'

# Basic model parameters as external flags.
FLAGS = None


# Randomizing again the dataset
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels)).astype(np.float32)
  labels = (np.arange(FLAGS.num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels


def main(_):
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

  # randomize the data
  train_dataset, train_labels = randomize(train_dataset, train_labels)
  test_dataset, test_labels = randomize(test_dataset, test_labels)
  print('Randomizing is done')
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


  # Reformat the data: image data as a cube (width by height by number 
  # of channels) and labels as float 1-hot encodings.
  train_dataset, train_labels = reformat(train_dataset, train_labels)
  test_dataset, test_labels = reformat(test_dataset, test_labels)
  print('Reformatted training set', train_dataset.shape, train_labels.shape)
  print('Reformatted test set', test_dataset.shape, test_labels.shape)

  def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

  def accuracy2(predictions, labels):
    print ('shape = ', np.argmax(predictions, 1).shape)
    sh = np.argmax(predictions, 1).shape
    predArg = np.argmax(predictions, 1)
    labArg = np.argmax(labels, 1)  
    count = 0
    pred = []
    lab = []
    print ('sh[0]', sh[0])
    
    for i in range(sh[0]):
      #print (i)
      if predArg[i] != labArg[i]:
        count = count + 1
        #print ('pred = ', predArg[i])
        #print ('lab = ', labArg[i]) 
        pred.append(predArg[i])
        lab.append(labArg[i])
        #print ('prediction =', predictions[i,:])
        #print ('labels =', labels[i,:])  
    print ('count =', count)  
    print ('pred =', pred)
    print ('lab =', lab)
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

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
    tf_train_labels = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.num_labels))
    tf_test_dataset = tf.constant(test_dataset)
  
    # Variables.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])

    W_conv4 = weight_variable([5, 5, 128, 256])
    b_conv4 = bias_variable([256])

    W_fc1 = weight_variable([28 * 28 * 256, 1024])
    b_fc1 = bias_variable([1024])

    W_fc2 = weight_variable([1024, 2048])
    b_fc2 = bias_variable([2048])

    W_fc3 = weight_variable([2048, FLAGS.num_labels])
    b_fc3 = bias_variable([FLAGS.num_labels])

  
    # Model.
    def model(data):
      h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
      h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
      #h_pool1 = max_pool_2x2(h_conv2)
      h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
      h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4) 
      #h_pool2 = max_pool_2x2(h_conv4)
      shape = h_conv4.get_shape().as_list()
      reshape = tf.reshape(h_conv4, [shape[0], shape[1] * shape[2] * shape[3]])
      h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)
      h_fc2 = tf.nn.tanh(tf.matmul(h_fc1, W_fc2) + b_fc2)
      return tf.matmul(h_fc2, W_fc3) + b_fc3
   
  
    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
    # Optimizer.
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

    step_plot = []
    loss_plot = []
    accuracy_plot = []

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
        acc = accuracy(predictions, batch_labels)
        print('Minibatch accuracy: %.1f%%' % acc)
        step_plot.append(step)
        loss_plot.append(l)
        accuracy_plot.append(acc)
    print('Test accuracy: %.1f%%' % accuracy2(test_prediction.eval(session=sess), test_labels))
    plt.figure(facecolor='white')
    plt.plot(step_plot, loss_plot, 'rs--')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss plot')
    plt.grid(True)
    plt.show()
    plt.figure(facecolor='white')
    plt.plot(step_plot, accuracy_plot, '--bs')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title('Accuracy plot')
    plt.grid(True)
    plt.show()
    print(accuracy_plot)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.005,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
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
  parser.add_argument(
      '--num_images_Train',
      type=int,
      default=10000,
      help='Number of images per label in train floder'
  )
  parser.add_argument(
      '--num_images_Test',
      type=int,
      default=100,
      help='Number of images per label in test floder'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=2601,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--num_labels',
      type=int,
      default=26,
      help='Number of labels'
  )

 
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  


