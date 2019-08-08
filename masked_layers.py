# Copyright (c) 2019 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras import backend as K
import numpy as np
from tf_plus import Conv2D, Dense


class MaskedDense(Dense):
    def __init__(self, units, mask_weight, mask_bias,
                 *args, **kwargs):
        super(MaskedDense, self).__init__(units, *args, **kwargs)        
        self.mask_weight = mask_weight
        self.mask_bias = mask_bias
        
    def build(self, input_shape):
        super(MaskedDense, self).build(input_shape)
        self._underlying_kernel = self.kernel
        self.kernel_mask = K.variable(value=self.mask_weight)
        self._non_trainable_weights.append(self.kernel_mask)
        self.kernel = self._underlying_kernel * self.kernel_mask
        
        if self.use_bias:
            self._underlying_bias = self.bias
            self.bias_mask = K.variable(value=self.mask_bias)
            self._non_trainable_weights.append(self.bias_mask)
            self.bias = self._underlying_bias * self.bias_mask
        else:
            self.bias = None
        

class MaskedConv2D(Conv2D):
    def __init__(self, filters, kernel_size, mask_weight, mask_bias,
                 *args, **kwargs):
        super(MaskedConv2D, self).__init__(filters, kernel_size, *args, **kwargs)        
        self.mask_weight = mask_weight
        self.mask_bias = mask_bias
        
    def build(self, input_shape):
        super(MaskedConv2D, self).build(input_shape)
        self._underlying_kernel = self.kernel
        self.kernel_mask = K.variable(value = self.mask_weight)
        self._non_trainable_weights.append(self.kernel_mask)
        self.kernel = self._underlying_kernel * self.kernel_mask
        
        if self.use_bias:
            self._underlying_bias = self.bias
            self.bias_mask = K.variable(value=self.mask_bias)
            self._non_trainable_weights.append(self.bias_mask)
            self.bias = self._underlying_bias * self.bias_mask
        else:
            self.bias = None

class FreezeDense(Dense):
    def __init__(self, units, init_weight, init_bias, mask_weight, mask_bias,
                 *args, **kwargs):
        super(FreezeDense, self).__init__(units, *args, **kwargs)        
        self.init_weight = init_weight
        self.init_bias = init_bias
        
        self.mask_weight = mask_weight
        self.mask_weight_rev = np.abs(mask_weight - 1)
        self.mask_bias = mask_bias
        self.mask_bias_rev = np.abs(mask_bias - 1)
        
    def build(self, input_shape):
        super(FreezeDense, self).build(input_shape)
        self._underlying_kernel = self.kernel
        
        self.kernel_init = K.variable(value = self.init_weight)
        self.kernel_mask = K.variable(value = self.mask_weight)
        self.kernel_mask_rev = K.variable(value = self.mask_weight_rev)
        
        self._non_trainable_weights.append(self.kernel_init)
        self._non_trainable_weights.append(self.kernel_mask)
        self._non_trainable_weights.append(self.kernel_mask_rev)
        
        self.kernel = self._underlying_kernel * self.kernel_mask + self.kernel_mask_rev * self.kernel_init
        
        if self.use_bias:
            self._underlying_bias = self.bias
            
            self.bias_init = K.variable(value = self.init_bias)
            self.bias_mask = K.variable(value=self.mask_bias)
            self.bias_mask_rev = K.variable(value = self.mask_bias_rev)
            
            self._non_trainable_weights.append(self.bias_init)
            self._non_trainable_weights.append(self.bias_mask)
            self._non_trainable_weights.append(self.bias_mask_rev)
            
            self.bias = self._underlying_bias * self.bias_mask + self.bias_mask_rev * self.bias_init
        else:
            self.bias = None
        
        
        
class FreezeConv2D(Conv2D):
    def __init__(self, filters, kernel_size, init_weight, init_bias, mask_weight, mask_bias,
                 *args, **kwargs):
        super(FreezeConv2D, self).__init__(filters, kernel_size, *args, **kwargs)        
        self.init_weight = init_weight
        self.init_bias = init_bias
        
        self.mask_weight = mask_weight
        self.mask_weight_rev = np.abs(mask_weight - 1)
        self.mask_bias = mask_bias
        self.mask_bias_rev = np.abs(mask_bias - 1)
        
    def build(self, input_shape):
        super(FreezeConv2D, self).build(input_shape)
        self._underlying_kernel = self.kernel
        
        self.kernel_init = K.variable(value = self.init_weight)
        self.kernel_mask = K.variable(value = self.mask_weight)
        self.kernel_mask_rev = K.variable(value = self.mask_weight_rev)
        
        self._non_trainable_weights.append(self.kernel_init)
        self._non_trainable_weights.append(self.kernel_mask)
        self._non_trainable_weights.append(self.kernel_mask_rev)
        
        self.kernel = self._underlying_kernel * self.kernel_mask + self.kernel_mask_rev * self.kernel_init
        
        if self.use_bias:
            self._underlying_bias = self.bias
            
            self.bias_init = K.variable(value = self.init_bias)
            self.bias_mask = K.variable(value=self.mask_bias)
            self.bias_mask_rev = K.variable(value = self.mask_bias_rev)
            
            self._non_trainable_weights.append(self.bias_init)
            self._non_trainable_weights.append(self.bias_mask)
            self._non_trainable_weights.append(self.bias_mask_rev)
            
            self.bias = self._underlying_bias * self.bias_mask + self.bias_mask_rev * self.bias_init
        else:
            self.bias = None

