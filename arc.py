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
        with open(path.numpy(), 'r') as fp:
            examples = json.load(fp)
            train_examples = tf.convert_to_tensor([(_pad_example(example['input']), _pad_example(example['output'])) for example in examples['train']])
            test_examples = tf.convert_to_tensor([(_pad_example(example['input']), _pad_example(example['output'])) for example in examples['test']])
            feature = {
                'train_examples': tf.train.Feature(int64_list=tf.train.Int64List(value=tf.reshape(train_examples, [-1]))),
                'train_examples_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=train_examples.shape)),
                'test_examples': tf.train.Feature(int64_list=tf.train.Int64List(value=tf.reshape(tf.constant(test_examples), [-1]))),
                'test_examples_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=test_examples.shape))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            return example.SerializeToString()

    def _parse_json_example_ds(path):
        example = tf.py_function(_parse_json_example, [path], tf.string)
        feature = {
            'train_examples': tf.io.VarLenFeature(tf.int64),
            'train_examples_shape': tf.io.FixedLenFeature([5], tf.int64),
            'test_examples': tf.io.VarLenFeature(tf.int64),
            'test_examples_shape': tf.io.FixedLenFeature([5], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature)
        train_examples = tf.reshape(tf.sparse.to_dense(example['train_examples']), example['train_examples_shape'])
        test_examples = tf.reshape(tf.sparse.to_dense(example['test_examples']), example['test_examples_shape'])
        return {
            'train': train_examples,
            'test': test_examples
        }

    def _create_arc_dataset(dir_name):
        dataset = tf.data.Dataset.list_files(f'{dir_name}/*.json') 
        return dataset.map(_parse_json_example_ds)

    training = _create_arc_dataset(f'{dir_name}/training')
    evaluation = _create_arc_dataset(f'{dir_name}/evaluation')
    test = _create_arc_dataset(f'{dir_name}/test')
    
    return training, evaluation, test
