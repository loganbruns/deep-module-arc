
""" Deep ARC Model """

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LayerNormalization, LSTM
from tensorflow import keras
from tensorflow.keras import Model

class ArcModel(Model):
    def __init__(self):
        super(ArcModel, self).__init__()

        # Layers
        conv_channels = [24, 48, 96, 192, 384]
        self.conv_layers = [Conv2D(channel, 3, 2, padding='same', activation='relu') for channel in conv_channels]
        self.layernorm1 = LayerNormalization()
        self.lstm = LSTM(384)
        self.conv_transpose_layers = [Conv2DTranspose(channel, 3, 2, padding='same', activation='relu') for channel in reversed(conv_channels)]
        self.layernorm2 = LayerNormalization()
        self.conv_final = Conv2D(11, 3, padding='same', activation='softmax')

        # Loss
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        # self.optimizer = tf.keras.optimizers.Adam()
        self.optimizer = tfa.optimizers.AdamW(weight_decay=1e-4)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
        self.train_iou = tf.keras.metrics.MeanIoU(11, name='train_iou')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')
        self.test_iou = tf.keras.metrics.MeanIoU(11, name='test_iou')

    @tf.function
    def call(self, train_length, train_examples, test_input, test_output):
        train_examples = tf.reshape(train_examples, [train_examples.shape[0]] + [-1] + list(train_examples.shape[-3:]))
        test_input = tf.reshape(test_input, [test_input.shape[0]] + [-1] + list(test_input.shape[-3:]))

        embeddings = []
        for images in [train_examples, test_input]:
            for i in range(images.shape[1]):
                x = images[:,i,:,:,:]

                # Add mask to categories and one hot encode
                x = tf.cast(tf.reduce_sum(x, axis=-1), dtype=tf.int32)
                x = tf.one_hot(x, 11)

                for conv in self.conv_layers:
                    x = conv(x)
                x = self.layernorm1(x)
                embeddings.append(x)

        embeddings = tf.stack(embeddings, axis=1)
        embeddings_shape = list(embeddings.shape)
        embeddings = tf.reshape(embeddings, embeddings_shape[:2] + [-1])

        x = self.lstm(embeddings)

        x = tf.reshape(x, [-1] + embeddings_shape[2:])
        for conv_transpose in self.conv_transpose_layers:
            x = conv_transpose(x)
        x = self.layernorm2(x)
        x = tf.squeeze(self.conv_final(x))
        return x

    @tf.function
    def train_step(self, train_length, train_examples, test_input, test_output):
        with tf.GradientTape() as tape:
            predictions = self(train_length, train_examples, test_input, test_output)
            test_output = tf.squeeze(tf.cast(tf.reduce_sum(test_output, axis=-1), dtype=tf.int32))
            loss = self.loss_object(test_output, predictions)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self.train_loss(loss)
            self.train_acc.update_state(test_output, predictions)
            self.train_iou.update_state(test_output, tf.argmax(predictions, axis=-1))
            return predictions

    @tf.function
    def test_step(self, train_length, train_examples, test_input, test_output):
        predictions = self(train_length, train_examples, test_input, test_output)
        test_output = tf.squeeze(tf.cast(tf.reduce_sum(test_output, axis=-1), dtype=tf.int32))
        t_loss = self.loss_object(test_output, predictions)
        self.test_loss(t_loss)
        self.test_acc.update_state(test_output, predictions)
        self.test_iou.update_state(test_output, tf.argmax(predictions, axis=-1))
        return predictions



