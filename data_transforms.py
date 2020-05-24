
""" Dataset Transforms """

import multiprocessing

import tensorflow as tf


def _max_size_of_example(train_examples, test_input, test_output):
    """ Determine the maximum mask size across training examples, test input, and test output """

    max_y, max_x = [], []
    
    max_y.append(tf.math.reduce_max(tf.reduce_sum(train_examples[:,:,0,:,1], axis=2)))
    max_x.append(tf.math.reduce_max(tf.reduce_sum(train_examples[:,:,:,0,1], axis=2)))

    max_y.append(tf.math.reduce_max(tf.reduce_sum(test_input[:,0,:,1], axis=1)))
    max_x.append(tf.math.reduce_max(tf.reduce_sum(test_input[:,:,0,1], axis=1)))

    max_y.append(tf.math.reduce_max(tf.reduce_sum(test_output[:,0,:,1], axis=1)))
    max_x.append(tf.math.reduce_max(tf.reduce_sum(test_output[:,:,0,1], axis=1)))

    max_y = tf.math.reduce_max(max_y)
    max_x = tf.math.reduce_max(max_x)
    
    return max_y, max_x


def _roll_example(id, train_length, train_examples, test_input, test_output, y_shift, x_shift):
    """ Roll all images by specified y and x shifts """

    train_examples = tf.roll(train_examples, y_shift, axis=2)
    train_examples = tf.roll(train_examples, x_shift, axis=3)

    test_input = tf.roll(test_input, y_shift, axis=1)
    test_input = tf.roll(test_input, x_shift, axis=2)

    test_output = tf.roll(test_output, y_shift, axis=1)
    test_output = tf.roll(test_output, x_shift, axis=2)

    return id, train_length, train_examples, test_input, test_output


def _random_roll_example(id, train_length, train_examples, test_input, test_output, pad_length=32):
    """ Randomly roll provided example """

    y_max, x_max = _max_size_of_example(train_examples, test_input, test_output)
    y_shift = tf.random.uniform([1], 0, pad_length - y_max, dtype=tf.int32)[0]
    x_shift = tf.random.uniform([1], 0, pad_length - x_max, dtype=tf.int32)[0]

    return _roll_example(id, train_length, train_examples, test_input, test_output, y_shift, x_shift)


def random_roll_dataset(dataset):
    """ Randomly roll dataset """

    return dataset.map(
        lambda id, train_length, train_examples, test_input, test_output: _random_roll_example(id, train_length, train_examples, test_input, test_output),
        num_parallel_calls=multiprocessing.cpu_count()
    )

