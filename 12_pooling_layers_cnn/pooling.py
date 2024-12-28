import argparse

import numpy as np

from utils import check_output


def get_paddings(array, pool_size, pool_stride):
    """ 
    get padding sizes 
    args:
    - array [array]: input np array NxwxHxC
    - pool_size [int]: window size
    - pool_stride [int]: stride
    returns:
    - paddings [list[list]]: paddings in np.pad format
    """
    # the purpose of get_paddings is to calculate the padding needed to make the input array
    # divisible by the pool_stride and pool_size values.
    # divisible means that the remainder of the division is 0
    # pool_stride is the number of pixels the window moves each time
    # pool_size is the size of the window
    #######
    # w is the width of the image
    # h is the height of the image
    # the first _ is the number of images, and the last _ is the number of channel
    # channels can be the number of colors in the image
    _, w, h, _ = array.shape
    wpad = (w // pool_stride) * pool_stride + pool_size - w
    hpad = (h // pool_stride) * pool_stride + pool_size - h
    # the interpretation of the padding is as follows:
    # the first two values are the padding for the first dimension of the array
    # why [0,0]? because we don't want to add any padding to the number of images
    # which means, the number of rows to add to the top and bottom of the array

    # the second two values are the padding for the second dimension of the array
    # why [0,wpad]? because we want to add padding to the width of the image
    # which means, the number of columns to add to the left and right of the array

    # the third two values are the padding for the third dimension of the array
    # why [0,hpad]? because we want to add padding to the height of the image
    # which means, the number of channels to add to the front and back of the array

    # the fourth two values are the padding for the fourth dimension of the array
    # why [0,0]? because we don't want to add any padding to the number of channels
    # which means, the number of images to add to the front and back of the array
    return [[0, 0], [0, wpad], [0, hpad], [0, 0]]


def get_output_size(shape, pool_size, pool_stride):
    """ 
    given input shape, pooling window and stride, output shape 
    args:
    - shape [list]: input shape
    - pool_size [int]: window size
    - pool_stride [int]: stride
    returns
    - output_shape [list]: output array shape
    """
    # the purpose of get_output_size is to calculate the output size of the pooling layer
    # the output size means the size of the array after applying the pooling operation
    w = shape[1]
    h = shape[2]
    new_w = (w - pool_size) // pool_stride + 1
    new_h = (h - pool_size) // pool_stride + 1
    return [shape[0], int(new_w), int(new_h), shape[3]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and process tf files')
    parser.add_argument('-f', '--pool_size', required=True, type=int, default=3,
                        help='pool filter size')
    parser.add_argument('-s', '--stride', required=True, type=int, default=3,
                        help='stride size')
    args = parser.parse_args()

    input_array = np.random.rand(1, 224, 224, 16)
    pool_size = args.pool_size
    pool_stride = args.stride

    # padd the input layer
    paddings = get_paddings(input_array, pool_size, pool_stride)
    padded = np.pad(input_array, paddings, mode='constant', constant_values=0)

    # get output size
    output_size = get_output_size(padded.shape, pool_size, pool_stride)
    output = np.zeros(output_size)

    # IMPLEMENT THE POOLING CALCULATION
    check_output(output)

    # How to run:
    # Example: python pooling.py -f 3 -s 3