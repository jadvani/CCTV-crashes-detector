# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 00:04:39 2020

"""

import keras.backend as K
from keras.layers import Layer


class SineReLU(Layer):

    def __init__(self, epsilon=0.0025, **kwargs):
        super(SineReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.epsilon = K.cast_to_floatx(epsilon)

    def call(self, Z):
        m = self.epsilon * (K.sin(Z) - K.cos(Z))
        A = K.maximum(m, Z)
        return A

    def get_config(self):
        config = {'epsilon': float(self.epsilon)}
        base_config = super(SineReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape