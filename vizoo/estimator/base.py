#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements the the base class for density ratio estimators.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


class DensityRatioEstimator(object):
    """
    Input data points must be of shape `[..., n_particles, n_x]`, the last
    dimension is the feature axis.
    """
    def __init__(self):
        pass

    def rbf_kernel(self, x1, x2, kernel_width):
        return tf.exp(-tf.reduce_sum(tf.square(x1 - x2), axis=-1) /
                      (2 * kernel_width ** 2))

    def gram(self, x1, x2, kernel_width):
        # x1: [..., n1, n_x]
        # x2: [..., n2, n_x]
        # kernel_width: [...]
        # return: [..., n1, n2]
        x_row = tf.expand_dims(x1, -2)
        x_col = tf.expand_dims(x2, -3)
        kernel_width = tf.expand_dims(
            tf.expand_dims(kernel_width, -1), -1)
        return self.rbf_kernel(x_row, x_col, kernel_width)

    def heuristic_kernel_width(self, x_samples, x_basis):
        # x_samples: [..., n_samples, n_x]
        # x_basis: [..., n_basis, n_x]
        # return: [...]
        n_samples = tf.shape(x_samples)[-2]
        n_basis = tf.shape(x_basis)[-2]
        x_samples_expand = tf.expand_dims(x_samples, -2)
        x_basis_expand = tf.expand_dims(x_basis, -3)
        pairwise_dist = tf.sqrt(
            tf.reduce_sum(tf.square(x_samples_expand - x_basis_expand),
                          axis=-1))
        k = n_samples * n_basis // 2
        top_k_values = tf.nn.top_k(
            tf.reshape(pairwise_dist, [-1, n_samples * n_basis]), k=k).values
        kernel_width = tf.reshape(top_k_values[:, -1],
                                  tf.shape(x_samples)[:-2])
        # kernel_width = tf.Print(kernel_width, [kernel_width],
        #                         message="kernel_width: ")
        return tf.stop_gradient(kernel_width)

    def optimal_ratio(self, x, qx_samples, px_samples):
        raise NotImplementedError()
