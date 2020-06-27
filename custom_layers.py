import tensorflow as tf


class FrozenBatchNorm2D(tf.keras.layers.Layer):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps


    def build(self, input_shape):
        self.weight = self.add_weight(name='weight', shape=[input_shape[-1]],
                                      initializer='zeros', trainable=False)
        self.bias = self.add_weight(name='bias', shape=[input_shape[-1]],
                                    initializer='zeros', trainable=False)
        self.running_mean = self.add_weight(name='running_mean', shape=[input_shape[-1]],
                                            initializer='zeros', trainable=False)
        self.running_var = self.add_weight(name='running_var', shape=[input_shape[-1]],
                                           initializer='ones', trainable=False)


    def call(self, x):
        scale = self.weight * tf.math.rsqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        return x * scale + shift


    def compute_output_shape(self, input_shape):
        return input_shape
