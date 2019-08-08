# Copyright (c) 2019 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

from .util import *
from .tfutil import *
from .backend import learning_phase, batchnorm_learning_phase
from .network import BaseLayer, Layers, SequentialNetwork

from .core import Lambda, Activation
from .regularizers import l2reg

# Make some common stuff from TF available for easy import
import tensorflow as tf
Conv2D = tf.layers.Conv2D
MaxPooling2D = tf.layers.MaxPooling2D
Flatten = tf.layers.Flatten
Dense = tf.layers.Dense
he_normal = tf.keras.initializers.he_normal()    # Must call to prouduce initializer object
relu = tf.nn.relu
softmax = tf.nn.softmax
UpSampling2D = tf.keras.layers.UpSampling2D
AveragePooling2D = tf.keras.layers.AveragePooling2D
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Embedding = tf.keras.layers.Embedding
Dropout = tf.layers.Dropout
