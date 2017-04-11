from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d

'''
def autoNN(hx, hy):
    num_pixels = hx * hy
    # Building the encoder
    encoder = input_data(shape=[None, num_pixels])
    #encoder = fully_connected(encoder, 2048)
    encoder = fully_connected(encoder, 1)

    # Building the decoder
    #decoder = fully_connected(encoder, 2048)
    decoder = fully_connected(encoder, num_pixels)
    return decoder


'''
'''
encoder = input_data(shape=[None, 256, 256, 1])
encoder = conv_2d(encoder, 16, 3, strides=[1,2,2,1], activation='relu')
encoder = conv_2d(encoder, 32, 3, strides=[1,2,2,1], activation='relu')
encoder = conv_2d(encoder, 64, 3, strides=[1,2,2,1], activation='relu')
encoder = fully_connected(encoder, 1)


encoder = tflearn.conv_2d(encoder, 16, 3, activation='relu')
encoder = tflearn.max_pool_2d(encoder, 1)
encoder = tflearn.conv_2d(encoder, 8, 3, activation='relu')
decoder = tflearn.upsample_2d(encoder, 1)
decoder = tflearn.conv_2d(encoder, 1, 3, activation='relu')
'''
'''
def autoNN(hx, hy):
    num_pixels = hx * hy
    # Building the encoder
    encoder = input_data(shape=[None, hx, hy, 1])
    encoder = conv_2d(encoder, 16, 3, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    encoder = conv_2d(encoder, 8, 3, activation='relu')
    
    decoder = upsample_2d(encoder, 2)
    decoder = conv_2d(decoder, 1, 3, activation='relu')
    return decoder
'''
'''
def autoNN(hx, hy):
    num_pixels = hx * hy
    encoder = input_data(shape=[None, hx, hy, 1])
    encoder = conv_2d(encoder, 16, 3, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    encoder = conv_2d(encoder, 16, 3, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    encoder = conv_2d(encoder, 8, 3, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    
    decoder = conv_2d(encoder, 8, 3, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 16, 3, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 16, 3, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 1, 3, activation='relu')
    return decoder
'''


def autoNN(hx, hy):
    num_pixels = hx * hy
    encoder = input_data(shape=[None, hx, hy, 1])
    encoder = conv_2d(encoder, 32, 3, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    encoder = conv_2d(encoder, 32, 3, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    encoder = conv_2d(encoder, 16, 3, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    encoder = conv_2d(encoder, 8, 3, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    
    decoder = conv_2d(encoder, 8, 3, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 16, 3, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 32, 3, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 32, 3, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 1, 3, activation='relu')
    return decoder


