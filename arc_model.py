
""" Deep ARC Model """

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LayerNormalization, LSTM
from tensorflow import keras
from tensorflow.keras import Model

class ArcModel(Model):
    def __init__(self):
        super(ArcModel, self).__init__()

        # Layers
        self.conv1 = Conv2D(24, 3, 2, padding='same', activation='relu')
        self.conv2 = Conv2D(48, 3, 2, padding='same', activation='relu')
        self.conv3 = Conv2D(96, 3, 2, padding='same', activation='relu')
        self.conv4 = Conv2D(192, 3, 2, padding='same', activation='relu')
        self.conv5 = Conv2D(384, 3, 2, padding='same', activation='relu')
        self.layernorm1 = LayerNormalization()
        self.lstm = LSTM(384)
        self.deconv1 = Conv2DTranspose(384, 3, 2, padding='same', activation='relu')
        self.deconv2 = Conv2DTranspose(192, 3, 2, padding='same', activation='relu')
        self.deconv3 = Conv2DTranspose(96, 3, 2, padding='same', activation='relu')
        self.deconv4 = Conv2DTranspose(48, 3, 2, padding='same', activation='relu')
        self.deconv5 = Conv2DTranspose(24, 3, 2, padding='same', activation='relu')
        self.layernorm2 = LayerNormalization()
        self.conv_final = Conv2D(11, 3, padding='same', activation='softmax')

        # Loss
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()
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

                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = self.conv5(x)
                x = self.layernorm1(x)
                embeddings.append(x)

        embeddings = tf.stack(embeddings, axis=1)
        embeddings_shape = list(embeddings.shape)
        embeddings = tf.reshape(embeddings, embeddings_shape[:2] + [-1])

        x = self.lstm(embeddings)

        x = tf.reshape(x, [-1] + embeddings_shape[2:])
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
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



