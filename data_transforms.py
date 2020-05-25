
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


def _random_remap_example(id, train_length, train_examples, test_input, test_output):
    """ Randomly remap most colors on all images. (Except 0 and 1 to not interfere with mask) """

    table = tf.lookup.experimental.DenseHashTable(key_dtype=tf.int32,
                                 value_dtype=tf.int32,
                                 default_value=-1,
                                 empty_key=-2,
                                 deleted_key=-1)
    table.erase(tf.range(1))
    table.insert(tf.range(2, 10), tf.random.shuffle(tf.range(2, 10)))
    table.insert(tf.range(2), tf.range(2))

    train_example = tf.reshape(table.lookup(tf.reshape(train_examples, [-1])), tf.shape(train_examples))
    test_input = tf.reshape(table.lookup(tf.reshape(test_input, [-1])), tf.shape(test_input))
    test_output = tf.reshape(table.lookup(tf.reshape(test_output, [-1])), tf.shape(test_output))

    return id, train_length, train_examples, test_input, test_output


def random_remap_dataset(dataset):
    """ Randomly remap most colors in dataset """

    return dataset.map(
        lambda id, train_length, train_examples, test_input, test_output: _random_remap_example(id, train_length, train_examples, test_input, test_output)
    )


def _random_flip_example(id, train_length, train_examples, test_input, test_output):
    """ Randomly flip all images """

    flip_left_right = tf.equal(tf.random.uniform(maxval=2, dtype=tf.int32, shape=[]), 0)
    train_examples = tf.cond(flip_left_right,
            lambda: tf.reshape(tf.image.flip_left_right(tf.reshape(train_examples, [-1, 32, 32, 2])), tf.shape(train_examples)),
            lambda: train_examples)
    test_input = tf.cond(flip_left_right,
            lambda: tf.reshape(tf.image.flip_left_right(tf.reshape(test_input, [-1, 32, 32, 2])), tf.shape(test_input)),
            lambda: test_input)
    test_output = tf.cond(flip_left_right,
            lambda: tf.reshape(tf.image.flip_left_right(tf.reshape(test_output, [-1, 32, 32, 2])), tf.shape(test_output)),
            lambda: test_output)
    
    return id, train_length, train_examples, test_input, test_output


def random_flip_dataset(dataset):
    """ Randomly flip dataset """

    return dataset.map(
        lambda id, train_length, train_examples, test_input, test_output: _random_flip_example(id, train_length, train_examples, test_input, test_output),
        num_parallel_calls=multiprocessing.cpu_count()
    )


