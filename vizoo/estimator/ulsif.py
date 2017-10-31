#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements the optimal density ratio estimator by unconstrained least square
importance fitting (uLSIF).
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from .base import DensityRatioEstimator


class ULSIFEstimator(DensityRatioEstimator):
    """
    Input data points must be of shape `[..., n_particles, n_x]`, the last
    dimension is the feature axis.
    """
    def __init__(self, lambda_=0.001, min_ratio=None):
        self._lambda = lambda_
        self._min_ratio = min_ratio
        super(DensityRatioEstimator, self).__init__()

    def H(self, px_samples, x_basis, kernel_width):
        # px_samples: [..., n_p, n_x]
        # x_basis: [..., n_basis, n_x]
        n_p = tf.shape(px_samples)[-2]
        # phi_px: [..., n_p, n_basis]
        phi_px = self.gram(px_samples, x_basis, kernel_width)
        print("phi_px", phi_px.get_shape())
        rank = phi_px.get_shape().ndims
        assert rank is not None
        perm = np.arange(0, rank, dtype=np.int32)
        perm[-1], perm[-2] = perm[-2], perm[-1]
        # phi_px_t: [..., n_basis, n_p]
        phi_px_t = tf.transpose(phi_px, perm=perm)
        print("phi_px_t:", phi_px_t.get_shape())
        # H: [..., n_basis, n_basis]
        H = tf.matmul(phi_px_t, phi_px) / tf.to_float(n_p)
        print("H:", H.get_shape())
        return H

    def h(self, qx_samples, x_basis, kernel_width):
        # phi_qx: [..., n_q, n_basis]
        phi_qx = self.gram(qx_samples, x_basis, kernel_width)
        print("phi_qx:", phi_qx.get_shape())
        # h: [..., n_basis]
        h = tf.reduce_mean(phi_qx, -2)
        print("h:", h.get_shape())
        return h

    def optimal_alpha(self, qx_samples, px_samples, x_basis, kernel_width):
        n_basis = tf.shape(x_basis)[-2]
        H_ = self.H(px_samples, x_basis, kernel_width)
        h_ = self.h(qx_samples, x_basis, kernel_width)
        alpha = tf.matmul(
            tf.matrix_inverse(H_ + self._lambda * tf.eye(n_basis)),
            tf.expand_dims(h_, -1))
        # alpha: [..., n_basis]
        alpha = tf.squeeze(alpha, -1)
        print("alpha:", alpha.get_shape())
        return alpha

    def optimal_ratio(self, x, qx_samples, px_samples):
        # x: [..., m, n_x]
        # qx_samples: [..., n_q, n_x]
        # px_samples: [..., n_p, n_x]
        print('x:', x.get_shape())
        print('qx_samples:', qx_samples.get_shape())
        print('px_samples:', px_samples.get_shape())
        x_samples = tf.concat([qx_samples, px_samples], axis=-2)
        x_basis = qx_samples
        # kernel_width: [...]
        kernel_width = self.heuristic_kernel_width(x_samples, x_basis)
        print('kernel_width:', kernel_width.get_shape())
        # alpha: [..., n_basis]
        alpha = self.optimal_alpha(qx_samples, px_samples, x_basis,
                                   kernel_width)
        print('alpha:', alpha.get_shape())
        # phi_x: [..., m, n_basis]
        phi_x = self.gram(x, x_basis, kernel_width)
        print("phi_x:", phi_x.get_shape())
        # ratio: [..., m]
        ratio = tf.reduce_sum(tf.expand_dims(alpha, -2) * phi_x, -1)
        print('ratio:', ratio.get_shape())
        if self._min_ratio:
            ratio = tf.maximum(self._min_ratio, ratio)
        # ratio = tf.Print(ratio, [ratio], message="ratio: ", summarize=20)
        return ratio
