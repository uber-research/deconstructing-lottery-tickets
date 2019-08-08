# Copyright (c) 2019 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

def l2reg(l2_strength):
    if l2_strength == 0:
        return lambda x: tf.zeros(())
    return lambda x: l2_strength * tf.reduce_sum(tf.square(x))
