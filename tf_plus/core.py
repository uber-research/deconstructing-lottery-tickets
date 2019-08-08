from .network import BaseLayer
import tensorflow as tf

class Lambda(BaseLayer):
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self._fn = fn

    def call(self, inputs):
        return self._fn(inputs)

class Activation(BaseLayer):
    def __init__(self, activation):
        super(Activation, self).__init__()
        if activation == 'relu':
            self._activation = tf.nn.relu
        elif activation == 'softmax':
            self._activation = tf.nn.softmax
        else:
            raise Exception('{} activation not supported'.format(activation))

    def call(self, inputs):
        return self._activation(inputs)
