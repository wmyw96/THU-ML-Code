#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import numpy as np
import zhusuan as zs
from dataset import load_movielens1m_mapped_ptest
from tensorflow.contrib import layers
import random


def select_from_axis1(para, indices):
    gather_para = tf.transpose(para, perm=[1, 0, 2])
    gather_para = tf.gather(gather_para, indices)
    gather_para = tf.transpose(gather_para, perm=[1, 0, 2])
    return gather_para


def pmf(observed, n, m, D, n_particles, alpha_u,
        alpha_v, alpha_pred, alpha_bias, is_training):
    with zs.BayesianNet(observed=observed) as model:
        mu_z = tf.zeros(shape=[n, D])
        log_std_z = tf.ones(shape=[n, D]) * tf.log(alpha_u)
        z = zs.Normal('z', mu_z, logstd=log_std_z,
                      n_samples=n_particles, group_event_ndims=1)  # [K, n, D]
        mu_v = tf.zeros(shape=[m, D])
        log_std_v = tf.ones(shape=[m, D]) * tf.log(alpha_v)
        v = zs.Normal('v', mu_v, logstd=log_std_v,
                      n_samples=n_particles, group_event_ndims=1)  # [K, m, D]
        mu_bias = tf.zeros(shape=[m, 1])
        log_std_bias = tf.ones(shape=[m, 1]) * tf.log(alpha_bias)
        bias = zs.Normal('bias', mu_bias, log_std_bias,
                         n_samples=n_particles, group_event_ndims=1) # [K, m, 1]
        bs = tf.tile(tf.expand_dims(bias, axis=1), [1, n, 1, 1])
        bs = tf.squeeze(bs, axis=3)
        tv = tf.transpose(v, [0, 2, 1])
        pred_mu = tf.matmul(z, tv) + bs  # [K, n, m]
        r = zs.Normal('r', pred_mu, logstd=tf.log(alpha_pred))
    return model, pred_mu


def q_net(observed, r, mask, n, D, m, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as variational:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        mu_v = \
            tf.get_variable('q_mu_v', shape=[m, D],
                            initializer=tf.random_normal_initializer(0, 0.1))
        log_std_v = \
            tf.get_variable('q_log_std_v', shape=[m, 1],
                            initializer=tf.random_normal_initializer(0, 0.1))
        log_std_v = tf.tile(log_std_v, [1, D])
        mu_bias = \
            tf.get_variable('q_mu_bias', shape=[m, 1],
                            initializer=tf.random_normal_initializer(0, 0.1))
        log_std_bias = \
            tf.get_variable('q_log_std_bias', shape=[m, 1],
                            initializer=tf.random_normal_initializer(0, 0.1))
        v = zs.Normal('v', mu_v, logstd=log_std_v,
                      n_samples=n_particles, group_event_ndims=1)  # [K, m, D]
        bias = zs.Normal('bias', mu_bias, logstd=log_std_bias,
                         n_samples=n_particles, group_event_ndims=1)  # [K, m, 1]
        input_v = tf.tile(tf.expand_dims(v, 1), [1, n, 1, 1])  # [K, n, m, D]
        input_bias = tf.tile(tf.expand_dims(bias, 1), [1, n, 1, 1])
        input_r = tf.tile(tf.expand_dims(tf.expand_dims(r, 0), 3),
                          [n_particles, 1, 1, 1]) - input_bias
        input_i = tf.concat([input_v, input_r], 3)  # [K, n, m, D+1]
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        mask3 = tf.tile(tf.expand_dims(mask, 0), [n_particles, 1, 1])
        maskDp1 = tf.tile(tf.expand_dims(mask3, 3), [1, 1, 1, D+1])
        input_i = input_i * maskDp1
        lz_r = layers.fully_connected(
            input_i, 200, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        maskl200 = tf.tile(tf.expand_dims(mask3, 3), [1, 1, 1, 200])
        lz_r = lz_r * maskl200
        lz_r = layers.fully_connected(
            input_i, 100, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        maskl100 = tf.tile(tf.expand_dims(mask3, 3), [1, 1, 1, 100])
        lz_r = lz_r * maskl100
        lz_r = tf.reduce_sum(lz_r, 2) / tf.reduce_sum(maskl100, 2)
        lz_r = layers.fully_connected(
            lz_r, 200, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        z_mean = layers.fully_connected(lz_r, D, activation_fn=None)
        z_log_std = layers.fully_connected(lz_r, D, activation_fn=None)
        z = zs.Normal('z', z_mean, logstd=z_log_std, n_samples=None,
                      group_event_ndims=1)
    return variational


def get_traing_data(M, head, tail, idx_list, user_movie, user_movie_score):
    mask = []
    rating = []
    cc = 0
    for i in range(tail - head):
        user_id = idx_list[head + i]
        cur_mask = [0] * M
        cur_rating = [0] * M
        for j in range(len(user_movie[user_id])):
            cur_mask[user_movie[user_id][j]] = 1
            cur_rating[user_movie[user_id][j]] = \
                user_movie_score[user_id][j] - 3
            cc += 1
        mask.append(cur_mask)
        rating.append(cur_rating)
    mask_batch = np.array(mask)
    rating_batch = np.array(rating) / np.sqrt(2)  # enforce var to be 1
    return mask_batch, rating_batch, cc


def get_test_data(M, head, tail, user_movie, user_movie_score,
                  user_movie_test, user_movie_score_test):
    mask = []
    rating = []
    bmask = []
    brating = []
    for i in range(tail - head):
        user_id = head + i
        cur_mask = [0] * M
        cur_rating = [0] * M
        for j in range(len(user_movie[user_id])):
            cur_mask[user_movie[user_id][j]] = 1
            cur_rating[user_movie[user_id][j]] = \
                user_movie_score[user_id][j] - 3
        mask.append(cur_mask)
        rating.append(cur_rating)

        cur_bmask = [0] * M
        cur_brating = [0] * M
        for j in range(len(user_movie_test[user_id])):
            cur_bmask[user_movie_test[user_id][j]] = 1
            cur_brating[user_movie_test[user_id][j]] = \
                user_movie_score_test[user_id][j] - 3
        bmask.append(cur_bmask)
        brating.append(cur_brating)

    infer_mask = np.array(mask)
    infer_rating = np.array(rating) / np.sqrt(2)  # enforce var to be 1
    mask_batch = np.array(bmask)
    rating_batch = np.array(brating) / np.sqrt(2)
    return infer_mask, infer_rating, mask_batch, rating_batch


if __name__ == '__main__':
    np.random.seed(1234)
    tf.set_random_seed(1237)
    M, N, train_data, valid_data, test_data, user_movie, \
        user_movie_score, movie_user, movie_user_score, \
        user_movie_test, user_movie_score_test,\
        user_movie_valid, user_movie_score_valid, \
        = load_movielens1m_mapped_ptest('data/ml-1m.zip', valid_map=True)

    # set configurations and hyper parameters
    N_train = np.shape(train_data)[0]
    N_test = np.shape(test_data)[0]
    N_valid = np.shape(valid_data)[0]
    n_z = 30
    batch_size = 100
    test_batch_size = 100
    valid_batch_size = 100
    K = 10
    num_epochs = 1000
    learning_rate = 0.01
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    iters = (N + batch_size - 1) // batch_size
    test_iters = (N + test_batch_size - 1) // test_batch_size
    valid_iters = (N + valid_batch_size - 1) // valid_batch_size
    result_path = 'tmp/pmf_vi/'

    hp_alpha_u = 1.0
    hp_alpha_v = 1.0
    hp_alpha_pred = 0.1
    hp_alpha_bias = 1.0

    # Find non-trained files or peoples
    trained_movie = [False] * M
    trained_user = [False] * N
    for i in range(N_train):
        trained_user[train_data[i, 0]] = True
        trained_movie[train_data[i, 1]] = True
    us = 0
    vs = 0
    for i in range(N):
        us += trained_user[i]
    for j in range(M):
        vs += trained_movie[j]
    print('Untrained users = %d, untrained movied = %d' % (N - us, M - vs))
    trained_movie = tf.constant(trained_movie, dtype=tf.bool)
    trained_user = tf.constant(trained_user, dtype=tf.bool)

    # Define models
    n = tf.placeholder(tf.int32, shape=[], name='n')
    m = tf.placeholder(tf.int32, shape=[], name='m')
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    infer_mask = tf.placeholder(tf.float32, shape=[None, M], name='infer_mask')
    infer_rating = tf.placeholder(tf.float32, shape=[None, M],
                                  name='infer_rating')
    gen_mask = tf.placeholder(tf.float32, shape=[None, M], name='gen_mask')
    gen_rating = tf.placeholder(tf.float32, shape=[None, M], name='gen_rating')

    def log_joint(observed):
        model, _ = pmf(observed, n, M, n_z, n_particles, hp_alpha_u,
                       hp_alpha_v, hp_alpha_pred, hp_alpha_bias, is_training)
        log_pz, log_pv, log_pbias, log_pr = \
            model.local_log_prob(['z', 'v', 'bias', 'r'])
        log_pr = tf.reduce_sum(log_pr * gen_mask, axis=2)
        log_pr = tf.reduce_sum(log_pr, axis=1) * \
            N / tf.cast(n, dtype=tf.float32)
        log_pz = tf.reduce_sum(log_pz, axis=1) * \
            N / tf.cast(n, dtype=tf.float32)
        log_pv = tf.reduce_sum(log_pv, axis=1)
        log_pbias = tf.reduce_sum(log_pbias, axis=1)
        return log_pz + log_pv + log_pbias + log_pr  # [K, n]

    variational = q_net({}, infer_rating, infer_mask, n, n_z,
                        M, n_particles, is_training)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    qv_samples, log_qv = variational.query('v', outputs=True,
                                           local_log_prob=True)
    qbias_samples, log_qbias = variational.query('bias', outputs=True,
                                                 local_log_prob=True)
    log_qz = tf.reduce_sum(log_qz, axis=1) * \
        N / tf.cast(n, dtype=tf.float32)
    log_qv = tf.reduce_sum(log_qv, axis=1)
    log_qbias = tf.reduce_sum(log_qbias, axis=1)

    lower_bound = tf.reduce_mean(
        zs.sgvb(log_joint, {'r': gen_rating},
                {'z': [qz_samples, log_qz], 'v': [qv_samples, log_qv],
                 'bias': [qbias_samples, log_qbias]}, axis=0))

    # Importance sampling estimates of marginal log likelihood
    is_log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(log_joint, {'r': gen_rating},
                            {'z': [qz_samples, log_qz],
                             'v': [qv_samples, log_qv],
                             'bias': [qbias_samples, log_qbias]}, axis=0))
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    # Prediction and SE calculation
    _, pred = pmf({'z': qz_samples, 'v': qv_samples,
                   'bias': qbias_samples,
                   'r': gen_rating},
                  n, M, n_z, n_particles, hp_alpha_u, hp_alpha_v,
                  hp_alpha_pred, hp_alpha_bias, is_training)
    pred = tf.reduce_mean(pred, axis=0)
    error = (pred - gen_rating) * gen_mask
    se = tf.reduce_sum(error * error)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    saver = tf.train.Saver(max_to_keep=10)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):

            epoch_time = -time.time()
            ses = []
            idxes = range(N)
            random.shuffle(idxes)
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate

            for t in range(iters):
                l = t * batch_size
                r = min((t + 1) * batch_size, N)
                cur_batch_size = r - l
                tr_mask, tr_rating, cc = get_traing_data(M, l, r, idxes,
                                                         user_movie,
                                                         user_movie_score)
                _, __ = sess.run([infer, se],
                                 feed_dict={n: cur_batch_size,
                                            m: M,
                                            is_training: True,
                                            n_particles: K,
                                            infer_mask: tr_mask,
                                            infer_rating: tr_rating,
                                            gen_mask: tr_mask,
                                            gen_rating: tr_rating,
                                            learning_rate_ph: learning_rate})
                ses.append(__)
            epoch_time += time.time()
            print('Epoch {}({:.1f}s): rmse = {}'.format(
                epoch + 1, epoch_time, np.sqrt(np.sum(ses) / N_train * 2)))

            if (epoch + 1) % 10 == 0:
                test_se = []
                time_test = -time.time()
                for t in range(valid_iters):
                    l = t * valid_batch_size
                    r = min((t + 1) * valid_batch_size, N)
                    cur_batch_size = r - l
                    in_mask, in_rating, out_mask, out_rating = \
                        get_test_data(M, l, r, user_movie, user_movie_score,
                                      user_movie_valid, user_movie_score_valid)
                    __ = sess.run(se,
                                  feed_dict={n: cur_batch_size,
                                             m: M,
                                             is_training: False,
                                             n_particles: K * 5,
                                             infer_mask: in_mask,
                                             infer_rating: in_rating,
                                             gen_mask: out_mask,
                                             gen_rating: out_rating,
                                             learning_rate_ph: learning_rate})
                    test_se.append(__)
                time_test += time.time()
                print('>>> VALIDATION ({:.1f}s)'.format(time_test))
                print('>> Valid rmse = {}'.format(
                    (np.sqrt(np.sum(test_se) / N_valid * 2))))

            if (epoch + 1) % 10 == 0:
                test_se = []
                time_test = -time.time()
                for t in range(test_iters):
                    l = t * test_batch_size
                    r = min((t + 1) * test_batch_size, N)
                    cur_batch_size = r - l
                    in_mask, in_rating, out_mask, out_rating = \
                        get_test_data(M, l, r, user_movie, user_movie_score,
                                      user_movie_test, user_movie_score_test)
                    __ = sess.run(se,
                                  feed_dict={n: cur_batch_size,
                                             m: M,
                                             is_training: False,
                                             n_particles: K * 5,
                                             infer_mask: in_mask,
                                             infer_rating: in_rating,
                                             gen_mask: out_mask,
                                             gen_rating: out_rating,
                                             learning_rate_ph: learning_rate})
                    test_se.append(__)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test rmse = {}'.format(
                    (np.sqrt(np.sum(test_se) / N_test * 2))))