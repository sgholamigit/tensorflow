from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt

# Load pickle file contains training dataset and test dataset   
#pickle_file = 'train10step7Domain10000.pickle' # 26 classes (brightness step of 10), each image has 16 different level of brightness
#pickle_file = 'br7Simple1000.pickle'
#pickle_file = 'brightnessData.pickle'
#pickle_file = 'brSimpleclass30.pickle' # 26 classes (brightness step of 10), each image has 4 different level of brightness
#pickle_file = 'train7d10class10000.pickle'
#pickle_file = 'trainset7Simple10000.pickle'
#pickle_file = 'train5stepSimple10000.pickle' # 52 classes (brightness step of 5), each image has 4 different level of brightness
pickle_file = 'br6class14DReg.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

# Randomizing again the dataset
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

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
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  W_conv3 = weight_variable([5, 5, 64, 128])
  b_conv3 = bias_variable([128])

  W_conv4 = weight_variable([5, 5, 128, 256])
  b_conv4 = bias_variable([256])

  W_fc1 = weight_variable([7 * 7 * 256, 1024])
  b_fc1 = bias_variable([1024])

  W_fc2 = weight_variable([1024, 2048])
  b_fc2 = bias_variable([2048])

  W_fc3 = weight_variable([2048, num_labels])
  b_fc3 = bias_variable([num_labels])

  
  # Model.
  def model(data):
    h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    h_pool1 = max_pool_2x2(h_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4) 
    h_pool2 = max_pool_2x2(h_conv4)
    shape = h_pool2.get_shape().as_list()
    reshape = tf.reshape(h_pool2, [shape[0], shape[1] * shape[2] * shape[3]])
    h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)
    h_fc2 = tf.nn.tanh(tf.matmul(h_fc1, W_fc2) + b_fc2)
    #keep_prob = tf.placeholder(tf.float32)
    #h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    return tf.matmul(h_fc2, W_fc3) + b_fc3
   
  
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


num_steps = 1001
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
  #print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
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



