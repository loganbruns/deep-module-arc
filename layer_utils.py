
""" Layer Utilities """

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Layer

class ResConvBlock(Layer):
    def __init__(self, channels, filters):
        super(ResConvBlock, self).__init__()
        self.channels = channels
        self.filters = filters

    def build(self, input_shape):
        self.conv_1 = Conv2D(input_shape[-1], self.filters, 1, padding='same', activation='relu')
        self.conv_1.build(input_shape)
        self.conv_2 = Conv2D(self.channels, self.filters, 2, padding='same', activation='relu')
        self.conv_2.build(input_shape[:-1] + 2 * input_shape[-1])

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = tf.concat([inputs, x], -1)
        x = self.conv_2(x)
        return x

class ResConvTransposeBlock(Layer):
    def __init__(self, channels, filters):
        super(ResConvTransposeBlock, self).__init__()
        self.channels = channels
        self.filters = filters

    def build(self, input_shape):
        self.conv_1 = Conv2DTranspose(input_shape[-1], self.filters, 1, padding='same', activation='relu')
        self.conv_1.build(input_shape)
        self.conv_2 = Conv2DTranspose(self.channels, self.filters, 2, padding='same', activation='relu')
        self.conv_2.build(input_shape[:-1] + 2 * input_shape[-1])

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = tf.concat([inputs, x], -1)
        x = self.conv_2(x)
        return x

