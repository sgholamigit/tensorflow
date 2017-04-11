import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from six.moves import cPickle as pickle
import tensorflow as tf
import argparse


def maybe_extract(filename, FLAGS, force=False):
  #np.random.seed(133)
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

# Function to merge the data
def Merge_folders(data_folders, size_per_class,FLAGS):
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

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels


def reformat(dataset, labels, FLAGS):
  dataset = dataset.reshape(
    (-1, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels)).astype(np.float32)
  return dataset, labels

