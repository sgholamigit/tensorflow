from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d


def autoNN2(hx, hy):
    num_pixels = hx * hy
    encoder = input_data(shape=[None, hx, hy, 1])
    encoder = conv_2d(encoder, 64, 3, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    encoder = conv_2d(encoder, 32, 3, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    encoder = conv_2d(encoder, 32, 3, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    encoder = conv_2d(encoder, 16, 3, activation='relu')
    #encoder = max_pool_2d(encoder, 2)
    filter_size = encoder.get_shape()[2]
    number_filters = encoder.get_shape()[3]
    
    decoder = conv_2d(encoder, 16, 3, activation='relu')
    #decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 32, 3, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 32, 3, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 64, 3, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 1, 3, activation='relu')
    return encoder, decoder, filter_size, number_filters


