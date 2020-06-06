
""" Deep ARC Model """

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LayerNormalization, LSTM
from tensorflow import keras
from tensorflow.keras import Model

# from layer_utils import ResConvBlock, ResConvTransposeBlock

class ArcModel(Model):

    def __init__(self, step):
        super(ArcModel, self).__init__()

        # Layers
        # conv_channels = [24, 48, 96, 192, 384]
        # conv_channels = [6, 12, 24, 48, 96]
        conv_channels = [3, 6, 12, 24, 48]
        # conv_channels = [1, 2, 4, 8, 16]
        self.conv_layers = []
        self.layernorm1 = LayerNormalization()
        self.lstm = LSTM(2 * conv_channels[-1])
        self.layernorm2 = LayerNormalization()
        self.conv_transpose_layers = []
        self.layernorm3 = LayerNormalization()
        self.conv_final = Conv2D(11, 3, padding='same', activation='softmax')

        conv_features = 2
        for i in range(conv_features):
            feature = []
            for channel in conv_channels:
                feature.append(Conv2D(channel, 3, 2, padding='same', activation='relu'))
                # self.conv_layers.append(ResConvBlock(channel, 3))
            self.conv_layers.append(feature)

        for channel in reversed(conv_channels):
            self.conv_transpose_layers.append(Conv2DTranspose(channel, 3, 2, padding='same', activation='relu'))
            # self.conv_transpose_layers.append(ResConvTransposeBlock(channel, 3))

        # Loss
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        # self.optimizer = tf.keras.optimizers.Adam()
        # self.optimizer = tfa.optimizers.AdamW(learning_rate=5e-3, weight_decay=1e-4)
        schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            [55000, 300000],
            [1e-0, 1e-1, 1e-2]
        )
        self.optimizer = tfa.optimizers.LAMB(.001 * schedule(step))
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
        self.train_iou = tf.keras.metrics.MeanIoU(11, name='train_iou')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')
        self.test_iou = tf.keras.metrics.MeanIoU(11, name='test_iou')
        self.best_test_loss = tf.Variable(100., trainable=False)

    @tf.function
    def call(self, train_length, train_examples, test_input):
        train_examples = tf.reshape(train_examples, [train_examples.shape[0]] + [-1] + list(train_examples.shape[-3:]))
        test_input = tf.reshape(test_input, [test_input.shape[0]] + [-1] + list(test_input.shape[-3:]))

        # Add mask to categories and one hot encode
        test_input = tf.cast(tf.reduce_sum(test_input, axis=-1), dtype=tf.int32)
        test_input = tf.one_hot(test_input, 11)
        train_examples = tf.cast(tf.reduce_sum(train_examples, axis=-1), dtype=tf.int32)
        train_examples = tf.one_hot(train_examples, 11)

        embeddings = []
        mask = tf.squeeze(tf.sequence_mask(train_length, 10))
        mask = tf.concat([mask, tf.ones((mask.shape[0], 1), dtype=tf.bool)], -1)

        for images in [train_examples, test_input]:
            for i in range(images.shape[1]):
                features = []
                for feature in self.conv_layers:
                    x = images[:,i,:,:,:]

                    for conv in feature:
                        x = conv(x)
                    feature_shape = list(x.shape)
                    features.append(x)

                x = tf.concat(features, axis=-1)
                x = self.layernorm1(x)
                embeddings.append(x)

        embeddings.append(tf.zeros(embeddings[-1].shape))

        embeddings = tf.stack(embeddings, axis=1)
        embeddings_shape = list(embeddings.shape)
        embeddings = tf.reshape(embeddings, embeddings_shape[:1] + [int(embeddings_shape[1] / 2), -1])

        x = self.lstm(inputs=embeddings, mask=tf.cast(mask, tf.bool))

        x = tf.reshape(x, [-1] + feature_shape[1:3] + [2 * feature_shape[3]])
        x = self.layernorm2(x)
        for conv_transpose in self.conv_transpose_layers:
            x = conv_transpose(x)
        x = self.layernorm3(x)
        x = tf.concat([x, tf.squeeze(test_input)], -1)
        x = tf.squeeze(self.conv_final(x))
        return x

    @tf.function
    def train_step(self, train_length, train_examples, test_input, test_output):
        with tf.GradientTape() as tape:
            predictions = self(train_length, train_examples, test_input)
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
        predictions = self(train_length, train_examples, test_input)
        test_output = tf.squeeze(tf.cast(tf.reduce_sum(test_output, axis=-1), dtype=tf.int32))
        t_loss = self.loss_object(test_output, predictions)
        self.test_loss(t_loss)
        self.test_acc.update_state(test_output, predictions)
        self.test_iou.update_state(test_output, tf.argmax(predictions, axis=-1))
        return predictions

    def reset_metrics(self):
        self.train_loss.reset_states()
        self.train_acc.reset_states()
        self.train_iou.reset_states()
        self.test_loss.reset_states()
        self.test_acc.reset_states()
        self.test_iou.reset_states()

    def is_best_loss(self, test_loss):
        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            return True
        else:
            return False
