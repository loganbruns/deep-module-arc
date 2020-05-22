""" Abstract Reasoning Corpus Dataset Reader """

import tensorflow as tf
import json

def ArcDataset(dir_name):
    """ Create a dataset to read ARC dataset """

    def _pad_example(example):
        example = tf.constant(example)
        padded = tf.pad(example, [[0, 30-example.shape[0]], [0, 30-example.shape[1]]])
        mask = tf.pad(tf.ones(example.shape, tf.int32), [[0, 30-example.shape[0]], [0, 30-example.shape[1]]])
        return tf.stack([padded, mask], axis=2)

    def _parse_json_example(path):
        path = path.numpy()
        with open(path, 'r') as fp:
            examples = json.load(fp)
            id = str(path).split('/')[-1].split('.')[0]
            train_examples = tf.convert_to_tensor([(_pad_example(example['input']), _pad_example(example['output'])) for example in examples['train']])
            test_input = tf.convert_to_tensor([_pad_example(example['input']) for example in examples['test']])
            test_output = tf.convert_to_tensor([_pad_example(example['output']) for example in examples['test']])
            feature = {
                'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(id)])),
                'train_examples': tf.train.Feature(int64_list=tf.train.Int64List(value=tf.reshape(train_examples, [-1]))),
                'train_examples_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=train_examples.shape)),
                'test_input': tf.train.Feature(int64_list=tf.train.Int64List(value=tf.reshape(tf.constant(test_input), [-1]))),
                'test_input_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=test_input.shape)),
                'test_output': tf.train.Feature(int64_list=tf.train.Int64List(value=tf.reshape(tf.constant(test_output), [-1]))),
                'test_output_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=test_output.shape))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            return example.SerializeToString()

    def _parse_json_example_ds(path):
        example = tf.py_function(_parse_json_example, [path], tf.string)
        feature = {
            'id': tf.io.VarLenFeature(tf.string),
            'train_examples': tf.io.VarLenFeature(tf.int64),
            'train_examples_shape': tf.io.FixedLenFeature([5], tf.int64),
            'test_input': tf.io.VarLenFeature(tf.int64),
            'test_input_shape': tf.io.FixedLenFeature([4], tf.int64),
            'test_output': tf.io.VarLenFeature(tf.int64),
            'test_output_shape': tf.io.FixedLenFeature([4], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature)
        id = tf.sparse.to_dense(example['id'])[0]
        train_examples = tf.reshape(tf.sparse.to_dense(example['train_examples']), example['train_examples_shape'])
        train_examples = tf.cast(train_examples, tf.int32)
        test_input = tf.reshape(tf.sparse.to_dense(example['test_input']), example['test_input_shape'])
        test_input = tf.cast(test_input, tf.int32)
        test_output = tf.reshape(tf.sparse.to_dense(example['test_output']), example['test_output_shape'])
        test_output = tf.cast(test_output, tf.int32)
        return (id, train_examples, test_input, test_output)

    def _create_arc_dataset(dir_name):
        dataset = tf.data.Dataset.list_files(f'{dir_name}/*.json') 
        return dataset.map(_parse_json_example_ds)

    training = _create_arc_dataset(f'{dir_name}/training')
    evaluation = _create_arc_dataset(f'{dir_name}/evaluation')
    test = _create_arc_dataset(f'{dir_name}/test')
    
    return training, evaluation, test
