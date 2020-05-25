
""" Train Deep ARC Model """

import os
import time

import tensorflow as tf

from arc import ArcDataset
from arc_model import ArcModel
from data_transforms import random_roll_dataset, random_remap_dataset, random_flip_dataset, random_rotate_dataset

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('experiment_name', None, 'Name of experiment to train and run.')

flags.DEFINE_string('gpu', '0', 'GPU to use')

flags.DEFINE_integer('batch_size', 32, 'Batch size')

def main(unparsed_argv):
    """start main training loop"""

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    #tf.config.optimizer.set_jit(True)

    # Set up experiment dir
    experiment_dir = f'./experiments/{FLAGS.experiment_name}'
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    # Load model
    model = ArcModel()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=model.optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, f'{experiment_dir}/tf_ckpts', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # Set up experimental logging
    train_log_dir = f'{experiment_dir}/logs/train'
    test_log_dir = f'{experiment_dir}/logs/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)        

    # Load NYUv2 depth dataset
    train, val, _ = ArcDataset('data')

    train = random_roll_dataset(train.cache())
    train = random_remap_dataset(train)
    train = random_flip_dataset(train)
    train = random_rotate_dataset(train)
    val = random_roll_dataset(val.cache())
    val = random_remap_dataset(val)
    val = random_flip_dataset(val)
    val = random_rotate_dataset(val)

    train = train.repeat().shuffle(400).batch(FLAGS.batch_size).prefetch(4)
    val = val.batch(FLAGS.batch_size).prefetch(1)

    # Start training loop
    once_per_train = False
    starttime = time.time()
    startstep = int(ckpt.step)
        
    once_per_epoch = False
    for id, train_length, train_examples, test_input, test_output in train:
        if test_output.shape[1] > 1:
            print(f'WARNING: skipping multiple test predictions: {test_output.shape}')
            continue

        with train_summary_writer.as_default():
            predictions = model.train_step(train_length, train_examples, test_input, test_output)
            ckpt.step.assign_add(1)

            if not once_per_train:
                model.summary()
                once_per_train = True

            tf.summary.scalar('loss', model.train_loss.result(), step=int(ckpt.step))
            tf.summary.scalar('accuracy', model.train_acc.result(), step=int(ckpt.step))
            tf.summary.scalar('iou', model.train_iou.result(), step=int(ckpt.step))

            train_summary_writer.flush()
                
        if int(ckpt.step) % 100 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("Training loss {:1.2f}, accuracy {:1.2f}, IoU {:1.2f}".format(model.train_loss.result(), model.train_acc.result(), model.train_iou.result()))

        if int(ckpt.step) % 500 == 0:
            with test_summary_writer.as_default():
                for id, train_length, train_examples, test_input, test_output in val:
                    if test_output.shape[1] > 1:
                        print(f'WARNING: skipping multiple test predictions: {test_output.shape}')
                        continue
                    test_predictions = model.test_step(train_length, train_examples, test_input, test_output)

                print(f"{int(ckpt.step)}: test loss={model.test_loss.result()}, test accuracy={model.test_acc.result()}, test iou={model.test_iou.result()}")
                tf.summary.scalar('loss', model.test_loss.result(), step=int(ckpt.step))
                tf.summary.scalar('accuracy', model.test_acc.result(), step=int(ckpt.step))
                tf.summary.scalar('iou', model.test_iou.result(), step=int(ckpt.step))

                test_summary_writer.flush()
                
            template = 'Step {}, Loss: {}, Test Loss: {}, Sec/Iters: {}'
            print (template.format(int(ckpt.step),
                                   model.train_loss.result(),
                                   model.test_loss.result(),
                                   (time.time()-starttime)/(int(ckpt.step)-startstep)))
            starttime = time.time()
            startstep = int(ckpt.step)

            #tf.saved_model.save(model, f'{experiment_dir}/tf_model/{int(ckpt.step)}/')

if __name__ == '__main__':
    app.run(main)
