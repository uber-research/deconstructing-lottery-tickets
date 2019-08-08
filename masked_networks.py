from __future__ import print_function
from __future__ import division

from tf_plus import Conv2D, MaxPooling2D, Flatten, Dense, relu, softmax, Activation
from tf_plus import Layers, SequentialNetwork, l2reg
from tf_plus import learning_phase

# use tensorflow's version of keras, or else get version incompatibility errors
from tensorflow.python import keras as tfkeras
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np
#import tensorflow_probability as tfp
glorot_normal = tf.keras.initializers.glorot_normal()

'''
Methods to set up special network architectures with binary masks being the only things trained
'''

def build_fc_supermask(args):
    kwargs = {}
    if args.signed_constant:
        kwargs['signed_constant'] = True
        kwargs['const_multiplier'] = args.signed_constant_multiplier
    if args.dynamic_scaling:
        kwargs['dynamic_scaling'] = True
    
    return SequentialNetwork([
        Flatten(),
        MaskedDense(300, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, activation=relu, kernel_regularizer=l2reg(args.l2),
            name='fc_1', **kwargs),
        MaskedDense(100, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, activation=relu, kernel_regularizer=l2reg(args.l2),
            name='fc_2', **kwargs),
        MaskedDense(10, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, activation=None, kernel_regularizer=l2reg(args.l2),
            name='fc_3', **kwargs)
    ])

def build_conv2_supermask(args):
    kwargs = {}
    if args.signed_constant:
        kwargs['signed_constant'] = True
        kwargs['const_multiplier'] = args.signed_constant_multiplier
    if args.dynamic_scaling:
        kwargs['dynamic_scaling'] = True

    return SequentialNetwork([
        MaskedConv2D(64, 3, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, padding='same', kernel_regularizer=l2reg(args.l2),
            name='conv2D_1', **kwargs),
        Activation('relu'),
        MaskedConv2D(64, 3, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, padding='same', kernel_regularizer=l2reg(args.l2),
            name='conv2D_2', **kwargs),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        Flatten(),
        MaskedDense(256, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, activation=relu, kernel_regularizer=l2reg(args.l2),
            name='fc_1', **kwargs),
        MaskedDense(256, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, activation=relu, kernel_regularizer=l2reg(args.l2),
            name='fc_2', **kwargs),
        MaskedDense(10, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, activation=None, kernel_regularizer=l2reg(args.l2),
            name='fc_3', **kwargs)
    ])

def build_conv4_supermask(args):
    kwargs = {}
    if args.signed_constant:
        kwargs['signed_constant'] = True
        kwargs['const_multiplier'] = args.signed_constant_multiplier
    if args.dynamic_scaling:
        kwargs['dynamic_scaling'] = True

    return SequentialNetwork([
        MaskedConv2D(64, 3, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, padding='same', kernel_regularizer=l2reg(args.l2),
            name='conv2D_1', **kwargs),
        Activation('relu'),
        MaskedConv2D(64, 3, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, padding='same', kernel_regularizer=l2reg(args.l2),
            name='conv2D_2', **kwargs),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        MaskedConv2D(128, 3, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, padding='same', kernel_regularizer=l2reg(args.l2),
            name='conv2D_3', **kwargs),
        Activation('relu'),
        MaskedConv2D(128, 3, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, padding='same', kernel_regularizer=l2reg(args.l2),
            name='conv2D_4', **kwargs),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        Flatten(),
        MaskedDense(256, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, activation=relu, kernel_regularizer=l2reg(args.l2),
            name='fc_1', **kwargs),
        MaskedDense(256, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, activation=relu, kernel_regularizer=l2reg(args.l2),
            name='fc_2', **kwargs),
        MaskedDense(10, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, activation=None, kernel_regularizer=l2reg(args.l2),
            name='fc_3', **kwargs)
    ])


def build_conv6_supermask(args):
    kwargs = {}
    if args.signed_constant:
        kwargs['signed_constant'] = True
        kwargs['const_multiplier'] = args.signed_constant_multiplier
    if args.dynamic_scaling:
        kwargs['dynamic_scaling'] = True

    return SequentialNetwork([
        MaskedConv2D(64, 3, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, padding='same', kernel_regularizer=l2reg(args.l2),
            name='conv2D_1', **kwargs),
        Activation('relu'),
        MaskedConv2D(64, 3, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, padding='same', kernel_regularizer=l2reg(args.l2),
            name='conv2D_2', **kwargs),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        MaskedConv2D(128, 3, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, padding='same', kernel_regularizer=l2reg(args.l2),
            name='conv2D_3', **kwargs),
        Activation('relu'),
        MaskedConv2D(128, 3, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, padding='same', kernel_regularizer=l2reg(args.l2),
            name='conv2D_4', **kwargs),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        MaskedConv2D(256, 3, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, padding='same', kernel_regularizer=l2reg(args.l2),
            name='conv2D_5', **kwargs),
        Activation('relu'),
        MaskedConv2D(256, 3, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, padding='same', kernel_regularizer=l2reg(args.l2),
            name='conv2D_6', **kwargs),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        Flatten(),
        MaskedDense(256, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, activation=relu, kernel_regularizer=l2reg(args.l2),
            name='fc_1', **kwargs),
        MaskedDense(256, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, activation=relu, kernel_regularizer=l2reg(args.l2),
            name='fc_2', **kwargs),
        MaskedDense(10, kernel_initializer=glorot_normal, sigmoid_bias=args.sigmoid_bias,
            round_mask=args.round_mask, activation=None, kernel_regularizer=l2reg(args.l2),
            name='fc_3', **kwargs)
    ])

# used in any masked layer's call()
def get_effective_mask(self):
    if self.round_mask:
        # during train, clamp a random 50% to their rounded values, and backprop the other 50% directly
        # during test, clamp all of them to their rounded values
        which_to_clamp = tf.cond(learning_phase(),
            lambda: gen_math_ops.round(tf.random.uniform(self.kernel_mask.shape, minval=0, maxval=1)),
            lambda: tf.ones(self.kernel_mask.shape))
        binary_mask = gen_math_ops.round(tf.nn.sigmoid(self.kernel_mask))
    else:
        # during train, clamp all of them to 0's and 1's sampled by bernoulli and backprop the probabilities
        # during test, clamp all of them to their rounded values
        # actually, sample them too
        which_to_clamp = tf.ones(self.kernel_mask.shape)
        binary_mask = tf.cond(learning_phase(),
            lambda: tf.cast(tf.distributions.Bernoulli(probs=tf.nn.sigmoid(self.kernel_mask)).sample(), dtype=tf.float32)
                + tf.nn.sigmoid(self.kernel_mask)
                - tf.stop_gradient(tf.nn.sigmoid(self.kernel_mask)),
            lambda: tf.cast(tf.distributions.Bernoulli(probs=tf.nn.sigmoid(self.kernel_mask)).sample(), dtype=tf.float32))

    return which_to_clamp * binary_mask + (1 - which_to_clamp) * tf.nn.sigmoid(self.kernel_mask)

# used to make a signed constant kernel
def make_signed_consts(kernel, multiplier=1.0):
    '''Take a kernel tensor, change each value to a constant while respecting the original sign'''
    mean, var = tf.nn.moments(kernel, axes = [x for x in range(len(kernel.shape))])
    val = tf.sqrt(var)
#     val = tf.math.reduce_std(kernel)
    return tf.sign(kernel) * val * multiplier

class MaskedDense(Dense):
    # untrainable normal Dense layer
    # trainable mask, that is sigmoided (maybe squished) and then multiplied to Dense
    def __init__(self, units, sigmoid_bias=0, round_mask=False, signed_constant=False, const_multiplier=1, dynamic_scaling=False, *args, **kwargs):
        super(MaskedDense, self).__init__(units, *args, **kwargs)
        self._uses_learning_phase = True
        self.sigmoid_bias = sigmoid_bias # bias to add before rounding to adjust prune percentage
        self.round_mask = round_mask # round instead of bernoulli sampling
        self.signed_constant = signed_constant
        self.const_multiplier = const_multiplier
        self.dynamic_scaling = dynamic_scaling

    def build(self, input_shape):
        super(MaskedDense, self).build(input_shape)
        mask_init = tfkeras.initializers.Constant(self.sigmoid_bias)

        self._trainable_weights.remove(self.kernel)
        self._non_trainable_weights.append(self.kernel)
        if self.use_bias:
            self._trainable_weights.remove(self.bias)
            self._non_trainable_weights.append(self.bias)

        self.kernel_mask = tf.get_variable('mask',
                                           shape=self.kernel.shape,
                                           dtype=self.dtype,
                                           initializer=mask_init,
                                           trainable=True)
        self._trainable_weights.append(self.kernel_mask)
    
    # same as original call() except round some sample to {0, 1} based on a sample
    def call(self, inputs):
        if self.signed_constant:
            self.kernel = make_signed_consts(self.kernel, self.const_multiplier)

        effective_mask = get_effective_mask(self)
        effective_kernel = self.kernel * effective_mask
        
        if self.dynamic_scaling:
            self.ones_in_mask = tf.reduce_sum(effective_mask)
            self.multiplier = tf.div(tf.to_float(tf.size(effective_mask)), self.ones_in_mask)
            effective_kernel = self.multiplier * effective_kernel

        # original code from https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/keras/layers/core.py:
        inputs = ops.convert_to_tensor(inputs)
        outputs = gen_math_ops.mat_mul(inputs, effective_kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs



class MaskedConv2D(Conv2D):
    # untrainable original conv2d layer, trainable max
    def __init__(self, filters, kernel_size, sigmoid_bias=0, round_mask=False, signed_constant=False, const_multiplier=1, dynamic_scaling=False, *args, **kwargs):
        super(MaskedConv2D, self).__init__(filters, kernel_size, *args, **kwargs)
        self._uses_learning_phase = True
        self.sigmoid_bias = sigmoid_bias # bias to add before rounding to adjust prune percentage
        self.round_mask = round_mask # round instead of bernoulli sampling
        self.signed_constant = signed_constant
        self.const_multiplier = const_multiplier
        self.dynamic_scaling = dynamic_scaling

    def build(self, input_shape):
        super(MaskedConv2D, self).build(input_shape)
        mask_init = tfkeras.initializers.Constant(self.sigmoid_bias)

        self._trainable_weights.remove(self.kernel)
        self._non_trainable_weights.append(self.kernel)
        if self.use_bias:
            self._trainable_weights.remove(self.bias)
            self._non_trainable_weights.append(self.bias)

        self.kernel_mask = tf.get_variable('mask',
                                           shape=self.kernel.shape,
                                           dtype=self.dtype,
                                           initializer=mask_init,
                                           trainable=True)
        self._trainable_weights.append(self.kernel_mask)

    # same as original call() except apply binary mask
    def call(self, inputs):
        if self.signed_constant:
            self.kernel = make_signed_consts(self.kernel, self.const_multiplier)

        effective_mask = get_effective_mask(self)
        effective_kernel = self.kernel * effective_mask

        if self.dynamic_scaling:
            self.ones_in_mask = tf.reduce_sum(effective_mask)
            self.multiplier = tf.div(tf.to_float(tf.size(effective_mask)), self.ones_in_mask)
            effective_kernel = self.multiplier * effective_kernel

        # original code from https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py:
        outputs = self._convolution_op(inputs, effective_kernel)
        if self.use_bias:
            if self.data_format == 'channels_first':
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

