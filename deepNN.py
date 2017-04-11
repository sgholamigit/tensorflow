from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d
import tflearn
import tensorflow as tf
# from tflearn.data_preprocessing import ImagePreprocessing
'''
def deepNN(hx, hy, nch):
    # Real-time data preprocessing
    # img_prep = ImagePreprocessing()
    # img_prep.add_featurewise_zero_center()
    # img_prep.add_featurewise_stdnorm()
    network = input_data(shape=[None, hx, hy, nch])
                     # data_preprocessing=img_prep)

    for i in range(7):
        network = conv_2d(network, 64, 3, strides=[1,2,2,1], activation='relu')
        if i < 6:
            for j in range(2):
                network = conv_2d(network, 16, 4, activation='relu')

    network = fully_connected(network, 16, activation='relu')
    network = fully_connected(network, 1, activation='relu')
    return network
'''

def deepNN(hx, hy, nch):
    network = input_data(shape=[None, hx, hy, nch])
    b1 = tflearn.initializations.normal(shape=None)
    network = fully_connected(network, 32, bias=True, bias_init=b1, activation='relu')
    #b2 = tflearn.initializations.normal(shape=None)
    #network = fully_connected(network, 128, bias=True, bias_init=b2, activation='relu')
    #b3 = tflearn.initializations.normal(shape=None)
    #network = fully_connected(network, 1, bias=True, bias_init=b3, activation='relu')
    network = fully_connected(network, 1, activation='relu')
    return network


