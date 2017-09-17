#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import numpy as np
import zhusuan as zs
from dataset import load_movielens1m_mapped
from tensorflow.contrib import layers


def select_from_axis1(para, indices):
    gather_para = tf.transpose(para, perm=[1, 0, 2])
    gather_para = tf.gather(gather_para, indices)
    gather_para = tf.transpose(gather_para, perm=[1, 0, 2])
    return gather_para


def pmf(observed, n, m, D, n_particles, alpha_u, alpha_v, alpha_pred):
    with zs.BayesianNet(observed=observed) as model:
        mu_z = tf.zeros(shape=[n, D])
        log_std_z = tf.ones(shape=[n, D]) * tf.log(alpha_u)
        z = zs.Normal('z', mu_z, log_std_z,
                      n_samples=n_particles, group_event_ndims=1) # [K, n, D]
        mu_v = tf.zeros(shape=[m, D])
        log_std_v = tf.ones(shape=[m, D]) * tf.log(alpha_v)
        v = zs.Normal('v', mu_v, log_std_v,
                      n_samples=n_particles, group_event_ndims=1) # [K, m, D]
        pred_mu = tf.matmul(z, v, transpose_b=True)  # [K, n, m]
        r = zs.Normal('r', pred_mu, tf.log(alpha_pred))
    return model, pred_mu


def q_net(observed, r, mask, n, D, m, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as variational:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        mu_v = \
            tf.get_variable('q_mu_v', shape=[m, D],
                            initializer=tf.random_normal_initializer(0, 0.1))
        log_std_v = \
            tf.get_variable('q_log_std_v', shape=[m, D],
                            initializer=tf.random_normal_initializer(0, 0.1))
        v = zs.Normal('v', mu_v, log_std_v,
                      n_samples=n_particles, group_event_ndims=1)  # [K, m, D]
        input_v = tf.tile(tf.expand_dims(v, 1), [1, n, 1, 1])
        input_r = tf.tile(tf.expand_dims(tf.expand_dims(r, 0), 3),
                          [K, 1, 1, 1])
        input = tf.concat([input_v, input_r], 3)  # [K, n, m, D+1]
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        mask3 = tf.tile(tf.expand_dims(mask, 0), [K, 1, 1])
        maskDp1 = tf.tile(tf.expand_dims(mask3, 3), [1, 1, 1, D+1])
        input = input * maskDp1
        lz_r = layers.fully_connected(
            tf.to_float(input), 100, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        maskl100 = tf.tile(tf.expand_dims(mask3, 3), [1, 1, 1, 100])
        lz_r = lz_r * maskl100
        lz_r = tf.concat([tf.reduce_sum(lz_r, 2), tf.reduce_mean(lz_r, 2)],
                         axis=2)
        lz_r = layers.fully_connected(
            lz_r, 100, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        z_mean = layers.fully_connected(lz_r, D, activation_fn=None)
        z_log_std = layers.fully_connected(lz_r, D, activation_fn=None)
        z = zs.Normal('z', z_mean, logstd=z_log_std, n_samples=None,
                      group_event_ndims=1)
    return variational


if __name__ == '__main__':
    np.random.seed(1234)
    tf.set_random_seed(1237)
    M, N, train_data, valid_data, test_data, user_movie, \
        user_movie_score, movie_user, movie_user_score \
        = load_movielens1m_mapped('data/ml-1m.zip')

    # set configurations and hyper parameters
    N_train = np.shape(train_data)[0]
    N_test = np.shape(test_data)[0]
    D = 30
    batch_size = 10
    test_batch_size = 100000
    learning_rate = 0.005
    K = 8
    num_steps = 500
    iters = (N_train + batch_size - 1) // batch_size
    test_iters = (N_test + test_batch_size - 1) // test_batch_size
    result_path = 'tmp/pmf_vi/'

    hp_alpha_u = 1.0
    hp_alpha_v = 1.0
    hp_alpha_pred = 0.2

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

    # Define models for prediction
    true_rating = tf.placeholder(tf.float32, shape=[None, ],
                                 name='true_rating')
    true_rating_2d = tf.tile(tf.expand_dims(true_rating, 0), [K, 1])
    _, pred_rating = pmf({'u': U, 'v': V, 'r': true_rating_2d}, N, M, D, K,
                         select_u, select_v, alpha_u, alpha_v, alpha_pred)
    pred_rating = tf.reduce_mean(pred_rating, axis=0)
    error = pred_rating - true_rating
    rmse = tf.sqrt(tf.reduce_mean(error * error))

    # Define models for HMC
    n = tf.placeholder(tf.int32, shape=[], name='n')
    m = tf.placeholder(tf.int32, shape=[], name='m')

    def log_joint(observed):
        model, _ = pmf(observed, n, m, D, K, subselect_u,
                       subselect_v, alpha_u, alpha_v, alpha_pred)
        log_pu, log_pv = model.local_log_prob(['u', 'v'])  # [K, N], [K, M]
        log_pr = model.local_log_prob('r')  # [K, batch]
        log_pu = tf.reduce_sum(log_pu, axis=1)
        log_pv = tf.reduce_sum(log_pv, axis=1)
        log_pr = tf.reduce_sum(log_pr, axis=1)
        return log_pu + log_pv + log_pr

    saver = tf.train.Saver(max_to_keep=10)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(num_steps):
            test_rmse = []
            time_test = -time.time()
            for t in range(test_iters):
                ed_pos = min((t + 1) * test_batch_size, N_test + 1)
                n_batch = ed_pos - t * test_batch_size
                su = test_data[t * test_batch_size:ed_pos, 0]
                sv = test_data[t * test_batch_size:ed_pos, 1]
                tr = test_data[t * test_batch_size:ed_pos, 2]
                re = sess.run(rmse, feed_dict={select_u: su, select_v: sv,
                                               true_rating: tr,
                                               n: n_batch,
                                               m: n_batch})
                test_rmse.append(re)
            time_test += time.time()
            print('>>> TEST ({:.1f}s)'.format(time_test))
            print('>> Test rmse = {}'.format(
                (np.mean(test_rmse))))

            epoch_time = -time.time()
            for i in range(N):
                nv = len(user_movie[i])
                sv = np.array(user_movie[i])
                tr = np.array(user_movie_score[i])
                ssu = np.zeros([nv])
                ssv = np.array(range(nv))
                _ = sess.run(trans_cand_U, feed_dict={candidate_idx_u: [i]})
                _ = sess.run(sample_u_op, feed_dict={select_v: sv,
                                                     true_rating: tr,
                                                     subselect_u: ssu,
                                                     subselect_v: ssv,
                                                     n: 1,
                                                     m: nv})
                _ = sess.run(trans_us_cand[i])
            for i in range(M):
                nu = len(movie_user[i])
                su = np.array(movie_user[i])
                tr = np.array(movie_user_score[i])
                ssv = np.zeros([nu])
                ssu = np.array(range(nu))
                _ = sess.run(trans_cand_V, feed_dict={candidate_idx_v: [i]})
                _ = sess.run(sample_v_op, feed_dict={select_u: su,
                                                     true_rating: tr,
                                                     subselect_u: ssu,
                                                     subselect_v: ssv,
                                                     n: nu,
                                                     m: 1})
                _ = sess.run(trans_vs_cand[i])
            epoch_time += time.time()
            print('Step %d(%.1fs): Finished!' % (step, epoch_time))
