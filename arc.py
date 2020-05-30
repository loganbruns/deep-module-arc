""" Abstract Reasoning Corpus Dataset Reader """

import tensorflow as tf
import json

def ArcDataset(dir_name, pad_length=32, sequence_length=10):
    """ Create a dataset to read ARC dataset """

    def _pad_example(example, pad_length):
        example = tf.constant(example)
        padded = tf.pad(example, [[0, pad_length-example.shape[0]], [0, pad_length-example.shape[1]]])
        mask = tf.pad(tf.ones(example.shape, tf.int32), [[0, pad_length-example.shape[0]], [0, pad_length-example.shape[1]]])
        return tf.stack([padded, mask], axis=2)

    def _pad_sequence(examples, sequence_length, two=True):
        length = len(examples)
        if length > sequence_length:
            print(f'WARNING: {length} > sequence_length')
        for i in range(sequence_length-length):
            if two:
                examples.append((tf.zeros((pad_length, pad_length, 2), dtype=tf.int32), tf.zeros((pad_length, pad_length, 2), dtype=tf.int32)))
            else:
                examples.append(tf.zeros((pad_length, pad_length, 2), dtype=tf.int32))
        examples = tf.convert_to_tensor(examples)
        return length, examples

    def _parse_json_example(path):
        path = path.numpy()
        with open(path, 'r') as fp:
            examples = json.load(fp)
            
            id = str(path).split('/')[-1].split('.')[0]
            
            train_examples = [(_pad_example(example['input'], pad_length), _pad_example(example['output'], pad_length)) for example in examples['train']]
            train_length, train_examples = _pad_sequence(train_examples, sequence_length)
            
            test_input = [_pad_example(example['input'], pad_length) for example in examples['test']]
            # test_input_length, test_input = _pad_sequence(test_input, sequence_length, two=False)
            test_input = tf.convert_to_tensor(test_input)

            test_output = [_pad_example(example['output'], pad_length) for example in examples['test']]
            if len(test_output) > 1:
                print(f'WARNING: ignoring {id} due to longer test sequence')
            else:
                print(f'INFO: processed {id}')
            # test_output_length, test_output = _pad_sequence(test_output, sequence_length, two=False)
            test_output = tf.convert_to_tensor(test_output)

            feature = {
                'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(id)])),
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

    def _parse_json_example_ds(path):
        example = tf.py_function(_parse_json_example, [path], tf.string)
        feature = {
            'id': tf.io.VarLenFeature(tf.string),
            'train_examples': tf.io.VarLenFeature(tf.int64),
            'train_examples_shape': tf.io.FixedLenFeature([5], tf.int64),
            'train_length': tf.io.FixedLenFeature([1], tf.int64),
            'test_input': tf.io.VarLenFeature(tf.int64),
            'test_input_shape': tf.io.FixedLenFeature([4], tf.int64),
            'test_output': tf.io.VarLenFeature(tf.int64),
            'test_output_shape': tf.io.FixedLenFeature([4], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature)
        id = tf.sparse.to_dense(example['id'])[0]
        train_examples = tf.reshape(tf.sparse.to_dense(example['train_examples']), example['train_examples_shape'])
        train_examples = tf.cast(train_examples, tf.int32)
        train_length = example['train_length']
        test_input = tf.reshape(tf.sparse.to_dense(example['test_input']), example['test_input_shape'])
        test_input = tf.cast(test_input, tf.int32)
        test_output = tf.reshape(tf.sparse.to_dense(example['test_output']), example['test_output_shape'])
        test_output = tf.cast(test_output, tf.int32)
        return (id, train_length, train_examples, test_input, test_output)

    def _create_arc_dataset(dir_name):
        dataset = tf.data.Dataset.list_files(f'{dir_name}/*.json')
        # dataset = dataset.map(_parse_json_example_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(_parse_json_example_ds)
        # DEBUG: for now ignore test results longer than 1
        dataset = dataset.filter(lambda id, train_length, train_examples, test_input, test_output: tf.math.equal(tf.shape(test_output)[0], 1))
        return dataset

    training = _create_arc_dataset(f'{dir_name}/training')
    evaluation = _create_arc_dataset(f'{dir_name}/evaluation')
    test = _create_arc_dataset(f'{dir_name}/test')
    
    return training, evaluation, test


def CompressedArcDataset(dir_name):
    """ Create a dataset to read ARC dataset """

    def _parse_example(example):
        feature = {
            'id': tf.io.VarLenFeature(tf.string),
            'train_examples': tf.io.VarLenFeature(tf.int64),
            'train_examples_shape': tf.io.FixedLenFeature([5], tf.int64),
            'train_length': tf.io.FixedLenFeature([1], tf.int64),
            'test_input': tf.io.VarLenFeature(tf.int64),
            'test_input_shape': tf.io.FixedLenFeature([4], tf.int64),
            'test_output': tf.io.VarLenFeature(tf.int64),
            'test_output_shape': tf.io.FixedLenFeature([4], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature)
        id = tf.sparse.to_dense(example['id'])[0]
        train_examples = tf.reshape(tf.sparse.to_dense(example['train_examples']), example['train_examples_shape'])
        train_examples = tf.cast(train_examples, tf.int32)
        train_length = example['train_length']
        test_input = tf.reshape(tf.sparse.to_dense(example['test_input']), example['test_input_shape'])
        test_input = tf.cast(test_input, tf.int32)
        test_output = tf.reshape(tf.sparse.to_dense(example['test_output']), example['test_output_shape'])
        test_output = tf.cast(test_output, tf.int32)
        return (id, train_length, train_examples, test_input, test_output)

    def _create_arc_dataset(dir_name):
        dataset = tf.data.Dataset.list_files(f'{dir_name}/*.tfrecord')
        dataset = dataset.interleave(lambda filename: tf.data.TFRecordDataset(filename, compression_type='GZIP'))
        dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    training = _create_arc_dataset(f'{dir_name}/training')
    evaluation = _create_arc_dataset(f'{dir_name}/evaluation')
    test = _create_arc_dataset(f'{dir_name}/test')
    
    return training, evaluation, test

