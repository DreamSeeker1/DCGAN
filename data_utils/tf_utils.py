"""process the pictures using Dataset API"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib as contrib
import os


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [64, 64])
    return image_resized


def get_pics(data_path):
    """Process the picture using Dataset.
    Args:
        data_path: path of the pictures folder
    Returns:
        dataset: the pics that have been resized
    """
    try:
        pic_list = os.listdir(data_path)
    except ValueError:
        print('Path not found')
        return
    filename = tf.constant(list(map(lambda x: os.path.join(data_path, x), pic_list)))
    dataset = tf.data.Dataset.from_tensor_slices(filename)
    dataset = dataset.map(_parse_function)
    return dataset
