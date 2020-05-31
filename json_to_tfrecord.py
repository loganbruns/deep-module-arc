
""" Convert ARC json to ARC tfrecord """

import tensorflow as tf

from arc import ArcDataset

def main():

    def serialize_example(id, train_length, train_examples, test_input, test_output):
        feature = {
            'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(id.numpy())])),
            'train_examples': tf.train.Feature(int64_list=tf.train.Int64List(value=tf.reshape(train_examples, [-1]))),
            'train_examples_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=train_examples.shape)),
            'train_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[train_length])),
            'test_input': tf.train.Feature(int64_list=tf.train.Int64List(value=tf.reshape(tf.constant(test_input), [-1]))),
            'test_input_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=test_input.shape)),
            'test_output': tf.train.Feature(int64_list=tf.train.Int64List(value=tf.reshape(tf.constant(test_output), [-1]))),
            'test_output_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=test_output.shape))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def tf_serialize_example(id, train_length, train_examples, test_input, test_output):
        tf_string = tf.py_function(
            serialize_example,
            (id, train_length, train_examples, test_input, test_output),
            tf.string)
        return tf.reshape(tf_string, ())

    train, val, test = ArcDataset('data')

    for dataset, filename in zip([train, val, test], ['training', 'evaluation', 'test']):
        serialized_dataset = dataset.map(tf_serialize_example)

        writer = tf.data.experimental.TFRecordWriter(f'data/{filename}/{filename}.tfrecord', compression_type='GZIP')
        writer.write(serialized_dataset)

if __name__ == '__main__':
    main()
