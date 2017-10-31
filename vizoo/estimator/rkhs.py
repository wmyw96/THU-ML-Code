#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements the optimal density ratio estimator in RKHS.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from .base import DensityRatioEstimator


class RKHSEstimator(DensityRatioEstimator):
    """
    Input data points must be of shape `[..., n_particles, n_x]`, the last
    dimension is the feature axis.
    """
    def __init__(self, lambda_=0.001, min_ratio=None):
        self._lambda = lambda_
        self._min_ratio = min_ratio
        super(DensityRatioEstimator, self).__init__()

    def optimal_alpha(self, qx_samples, px_samples, kernel_width):
        # qx_samples: [..., n_q, n_x]
        # px_samples: [..., n_p, n_x]
        n_p = tf.shape(px_samples)[-2]
        n_q = tf.shape(qx_samples)[-2]
        # Kp: [..., n_p, n_p]
        Kp = self.gram(px_samples, px_samples, kernel_width)
        print('Kp:', Kp.get_shape())
        # Kpq: [..., n_p, n_q]
        Kpq = self.gram(px_samples, qx_samples, kernel_width)
        print('Kpq:', Kpq.get_shape())
        # tmp: [..., n_p, n_p]
        tmp = 1. / tf.to_float(n_p) * Kp + self._lambda * tf.eye(n_p)
        # alpha: [..., n_p, n_q]
        alpha = -1. / (self._lambda * tf.to_float(n_p * n_q)) * tf.matmul(
            tf.matrix_inverse(tmp), Kpq)
        # alpha: [..., n_p]
        alpha = tf.reduce_sum(alpha, axis=-1)
        return alpha

    def optimal_ratio(self, x, qx_samples, px_samples):
        # x: [..., m, n_x]
        # qx_samples: [..., n_q, n_x]
        # px_samples: [..., n_p, n_x]
        print('x:', x.get_shape())
        print('qx_samples:', qx_samples.get_shape())
        print('px_samples:', px_samples.get_shape())
        x_samples = tf.concat([px_samples, qx_samples], axis=-2)
        kernel_width = self.heuristic_kernel_width(x_samples, x_samples)
        print('kernel_width:', kernel_width.get_shape())
        # alpha: [..., n_p]
        alpha = self.optimal_alpha(qx_samples, px_samples, kernel_width)
        print('alpha:', alpha.get_shape())
        # beta: []
        beta = 1. / (self._lambda * tf.to_float(tf.shape(qx_samples)[-2]))
        print('beta:', beta.get_shape())

        # Kxp: [..., m, n_p]
        Kxp = self.gram(x, px_samples, kernel_width)
        print('Kxp:', Kxp.get_shape())
        # Kxq: [..., m, n_q]
        Kxq = self.gram(x, qx_samples, kernel_width)
        print('Kxq:', Kxq.get_shape())
        # ratio: [..., m]
        ratio = tf.squeeze(tf.matmul(Kxp, tf.expand_dims(alpha, -1)), -1) + \
            tf.reduce_sum(Kxq * beta, axis=-1)
        print('ratio:', ratio.get_shape())
        if self._min_ratio:
            ratio = tf.maximum(self._min_ratio, ratio)
        # ratio: [..., m]
        # ratio = tf.Print(ratio, [ratio], message="ratio: ", summarize=20)
        return ratio
