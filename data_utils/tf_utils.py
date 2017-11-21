"""process the pictures using Dataset API"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib as contrib
import os
import model.params as mparams


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, mparams.channel)
    image_shape = tf.cast(tf.shape(image_decoded), dtype=tf.float32)
    image_cropped = tf.random_crop(image_decoded,
                                   size=tf.cast([image_shape[0] * 0.9, image_shape[1] * 0.9, image_shape[2]],
                                                dtype=tf.int32))
    image_resized = tf.image.resize_images(image_cropped, [64, 64]) / 255. * 2. - 1.
    return image_resized


def helper(path_list):
    """A small helper used to concat the path
    Args:
        path_list: a list of path
    Returns:
        concat_path: the concatenated path
    """
    concat_path = path_list[0]
    for i in path_list[1:]:
        concat_path = os.path.join(concat_path, i)
    return concat_path


def get_pics(data_path):
    """Process the picture using Dataset.
    Args:
        data_path: path of the pictures folder
    Returns:
        dataset: the pics that have been resized
    """
    pic_list = []
    try:
        tmp_list = list(map(lambda x: helper([data_path, x]), os.listdir(data_path)))
        for pic_folder in tmp_list:
            pic_list += list(map(lambda x: helper([pic_folder, x]), os.listdir(pic_folder)))
    except ValueError:
        print('Path not found')
        return
    filename = tf.constant(pic_list)
    dataset = tf.data.Dataset.from_tensor_slices(filename)
    dataset = dataset.map(_parse_function)
    return dataset
