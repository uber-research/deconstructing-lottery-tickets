from __future__ import print_function
from __future__ import division
import tensorflow as tf
from tf_plus import Conv2D, MaxPooling2D, Flatten, Dense, relu, Activation
from tf_plus import Layers, SequentialNetwork, l2reg
from masked_layers import MaskedDense, MaskedConv2D, FreezeDense, FreezeConv2D
# use tensorflow's version of keras, or else get version incompatibility errors
from tensorflow.python import keras as tfkeras
glorot_normal = tf.keras.initializers.glorot_normal()


'''
Methods to set up network architectures
'''


def build_fc_lottery(args):
    return SequentialNetwork([
        Flatten(),
#         BatchNormalization(momentum=0, name='batch_norm_1'),
        Dense(300, kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
        Dense(100, kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_2'),
        Dense(10, kernel_initializer=glorot_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_3')
    ])


def build_masked_fc_lottery(args, mask_values):
    return SequentialNetwork([
        Flatten(),
            # BatchNormalization(momentum=0, name='batch_norm_1'),
        MaskedDense(300, mask_values[0], mask_values[1],
                   kernel_initializer=glorot_normal, activation=relu, name='fc_1'),
        MaskedDense(100, mask_values[2], mask_values[3],
                   kernel_initializer=glorot_normal, activation=relu, name='fc_2'),
        MaskedDense(10, mask_values[4], mask_values[5],
                   kernel_initializer=glorot_normal, activation=None, name='fc_3')
    ])

def build_frozen_fc_lottery(args, init_values, mask_values):
    return SequentialNetwork([
        Flatten(),
            # BatchNormalization(momentum=0, name='batch_norm_1'),
        FreezeDense(300, init_values[0], init_values[1], mask_values[0], mask_values[1],
                   kernel_initializer=glorot_normal, activation=relu, name='fc_1'),
        FreezeDense(100, init_values[2], init_values[3], mask_values[2], mask_values[3],
                   kernel_initializer=glorot_normal, activation=relu, name='fc_2'),
        FreezeDense(10, init_values[4], init_values[5], mask_values[4], mask_values[5],
                   kernel_initializer=glorot_normal, activation=None, name='fc_3')
    ])



def build_conv2_lottery(args): 
    return SequentialNetwork([
        Conv2D(64, 3, kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_1'),
        Activation('relu'),
        Conv2D(64, 3, kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_2'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        Flatten(),
        Dense(256, kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
        Dense(256, kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_2'),
        Dense(10, kernel_initializer=glorot_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_3')
    ])

def build_masked_conv2_lottery(args, mask_values): 
    return SequentialNetwork([
        MaskedConv2D(64, 3, mask_values[0], mask_values[1], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_1'),
        Activation('relu'),
        MaskedConv2D(64, 3, mask_values[2], mask_values[3], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_2'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        Flatten(),
        MaskedDense(256, mask_values[4], mask_values[5], kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
        MaskedDense(256, mask_values[6], mask_values[7], kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_2'),
        MaskedDense(10, mask_values[8], mask_values[9], kernel_initializer=glorot_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_3')
    ])


def build_frozen_conv2_lottery(args, init_values, mask_values): 
    return SequentialNetwork([
        FreezeConv2D(64, 3, init_values[0], init_values[1], mask_values[0], mask_values[1], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_1'),
        Activation('relu'),
        FreezeConv2D(64, 3, init_values[2], init_values[3], mask_values[2], mask_values[3], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_2'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        Flatten(),
        FreezeDense(256, init_values[4], init_values[5], mask_values[4], mask_values[5], kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
        FreezeDense(256, init_values[6], init_values[7], mask_values[6], mask_values[7], kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_2'),
        FreezeDense(10, init_values[8], init_values[9], mask_values[8], mask_values[9], kernel_initializer=glorot_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_3')
    ])


def build_conv4_lottery(args): 
    return SequentialNetwork([
        Conv2D(64, 3, kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_1'),
        Activation('relu'),
        Conv2D(64, 3, kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_2'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        Conv2D(128, 3, kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_3'),
        Activation('relu'),
        Conv2D(128, 3, kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_4'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        Flatten(),
        Dense(256, kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
        Dense(256, kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_2'),
        Dense(10, kernel_initializer=glorot_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_3')
    ])


def build_masked_conv4_lottery(args, mask_values): 
    return SequentialNetwork([
        MaskedConv2D(64, 3, mask_values[0], mask_values[1], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_1'),
        Activation('relu'),
        MaskedConv2D(64, 3, mask_values[2], mask_values[3], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_2'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        MaskedConv2D(128, 3, mask_values[4], mask_values[5], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_3'),
        Activation('relu'),
        MaskedConv2D(128, 3, mask_values[6], mask_values[7], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_4'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        Flatten(),
#         Dropout(0.5),
        MaskedDense(256, mask_values[8], mask_values[9], kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
#         Dropout(0.5),
        MaskedDense(256, mask_values[10], mask_values[11], kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_2'),
#         Dropout(0.5),
        MaskedDense(10, mask_values[12], mask_values[13], kernel_initializer=glorot_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_3')
    ])


def build_frozen_conv4_lottery(args, init_values, mask_values): 
    return SequentialNetwork([
        FreezeConv2D(64, 3, init_values[0], init_values[1], mask_values[0], mask_values[1], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_1'),
        Activation('relu'),
        FreezeConv2D(64, 3, init_values[2], init_values[3], mask_values[2], mask_values[3], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_2'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        
        FreezeConv2D(128, 3, init_values[4], init_values[5], mask_values[4], mask_values[5], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_3'),
        Activation('relu'),
        FreezeConv2D(128, 3, init_values[6], init_values[7], mask_values[6], mask_values[7], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_4'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        
        Flatten(),
#         Dropout(0.5),
        FreezeDense(256, init_values[8], init_values[9], mask_values[8], mask_values[9], kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
#         Dropout(0.5),
        FreezeDense(256, init_values[10], init_values[11], mask_values[10], mask_values[11], kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_2'),
#         Dropout(0.5),
        FreezeDense(10, init_values[12], init_values[13], mask_values[12], mask_values[13], kernel_initializer=glorot_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_3')
    ])



def build_conv6_lottery(args): 
    return SequentialNetwork([
        Conv2D(64, 3, kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_1'),
        Activation('relu'),
        Conv2D(64, 3, kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_2'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        Conv2D(128, 3, kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_3'),
        Activation('relu'),
        Conv2D(128, 3, kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_4'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        Conv2D(256, 3, kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_5'),
        Activation('relu'),
        Conv2D(256, 3, kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_6'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        Flatten(),
#         Dropout(0.5),
        Dense(256, kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
#         Dropout(0.5),
        Dense(256, kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_2'),
#         Dropout(0.5),
        Dense(10, kernel_initializer=glorot_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_3')
    ])

def build_masked_conv6_lottery(args, mask_values): 
    return SequentialNetwork([
        MaskedConv2D(64, 3, mask_values[0], mask_values[1], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_1'),
        Activation('relu'),
        MaskedConv2D(64, 3, mask_values[2], mask_values[3], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_2'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        MaskedConv2D(128, 3, mask_values[4], mask_values[5], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_3'),
        Activation('relu'),
        MaskedConv2D(128, 3, mask_values[6], mask_values[7], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_4'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),   
        MaskedConv2D(256, 3, mask_values[8], mask_values[9], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_5'),
        Activation('relu'),
        MaskedConv2D(256, 3, mask_values[10], mask_values[11], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_6'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),   
        Flatten(),
#         Dropout(0.5),
        MaskedDense(256, mask_values[12], mask_values[13], kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
#         Dropout(0.5),
        MaskedDense(256, mask_values[14], mask_values[15], kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_2'),
#         Dropout(0.5),
        MaskedDense(10, mask_values[16], mask_values[17], kernel_initializer=glorot_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_3')
    ])


def build_frozen_conv6_lottery(args, init_values, mask_values): 
    return SequentialNetwork([
        FreezeConv2D(64, 3, init_values[0], init_values[1], mask_values[0], mask_values[1], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_1'),
        Activation('relu'),
        FreezeConv2D(64, 3, init_values[2], init_values[3], mask_values[2], mask_values[3], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_2'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        
        FreezeConv2D(128, 3, init_values[4], init_values[5], mask_values[4], mask_values[5], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_3'),
        Activation('relu'),
        FreezeConv2D(128, 3, init_values[6], init_values[7], mask_values[6], mask_values[7], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_4'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        
        FreezeConv2D(256, 3, init_values[8], init_values[9], mask_values[8], mask_values[9], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_5'),
        Activation('relu'),
        FreezeConv2D(256, 3, init_values[10], init_values[11], mask_values[10], mask_values[11], kernel_initializer=glorot_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_6'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        Flatten(),
#         Dropout(0.5),
        FreezeDense(256, init_values[12], init_values[13], mask_values[12], mask_values[13], kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
#         Dropout(0.5),
        FreezeDense(256, init_values[14], init_values[15], mask_values[14], mask_values[15], kernel_initializer=glorot_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_2'),
#         Dropout(0.5),
        FreezeDense(10, init_values[16], init_values[17], mask_values[16], mask_values[17], kernel_initializer=glorot_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_3')
    ])
