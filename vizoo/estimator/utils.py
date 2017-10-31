#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import zhusuan as zs


def adaptive_contrast_kl(estimator, qx_samples, px_samples, log_p_qx,
                         reverse_trick=True):
    """
    Estimate the exclusive KL divergence between q and p using density ratio
    estimator. q is a arbitrary distribution. p is a standard normal.

    By default KL(q||p) is calculated by -E_q log (p/q), and adaptive contrast
    is used.

    :param estimator: A density ratio estimator instance.
    :param qx_samples: Samples from q. Must be of shape [..., n_q, n_x].
    :param px_samples: Samples from p. Must be of shape [..., n_p, n_x].
    :param log_p_qx: log p(qx_samples). Must be of shape [..., n_q].

    :return: A Tensor of shape [...].
    """
    # [..., 1, n_x]
    qx_mu, qx_var = tf.nn.moments(qx_samples, axes=[-2], keep_dims=True)
    print('qx_mu:', qx_mu.get_shape())
    print('qx_var:', qx_var.get_shape())
    qx_std = tf.sqrt(qx_var)
    # [..., n_q, n_x]
    qx_tilde = (qx_samples - qx_mu) / qx_std
    print('qx_tilde:', qx_tilde.get_shape())
    rz = zs.distributions.Normal(qx_mu, tf.log(qx_std),
                                 is_reparameterized=False, group_ndims=1)
    # [..., n_q]
    log_r_qx = rz.log_prob(qx_samples)
    print('log_r_qx:', log_r_qx.get_shape())
    if reverse_trick:
        # [..., n_q]
        ratio_rq = estimator.optimal_ratio(
            qx_tilde, tf.stop_gradient(px_samples), tf.stop_gradient(qx_tilde))
        print('ratio_rq:', ratio_rq.get_shape())
        kl_term_qr = -tf.reduce_mean(tf.log(ratio_rq), axis=-1)
    else:
        ratio_qr = estimator.optimal_ratio(
            qx_tilde, tf.stop_gradient(qx_tilde), tf.stop_gradient(px_samples))
        kl_term_qr = tf.reduce_mean(tf.log(ratio_qr), axis=-1)

    # [...]
    kl_term = tf.reduce_mean(log_r_qx - log_p_qx, -1) + kl_term_qr
    print('kl_term:', kl_term.get_shape())
    return kl_term
