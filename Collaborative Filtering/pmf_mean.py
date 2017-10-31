#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Results:
# test 0.8412

import time
import tensorflow as tf
import numpy as np
import zhusuan as zs
from dataset import load_movielens1m_mapped


def select_from_axis1(para, indices):
    gather_para = tf.transpose(para, perm=[1, 0, 2])
    gather_para = tf.gather(gather_para, indices)
    gather_para = tf.transpose(gather_para, perm=[1, 0, 2])
    return gather_para


def pmf(observed, n, m, D, n_particles, select_u, select_v,
        alpha_u, alpha_v, alpha_pred):
    with zs.BayesianNet(observed=observed) as model:
        mu_u = tf.zeros(shape=[n, D])
        log_std_u = tf.ones(shape=[n, D]) * tf.log(alpha_u)
        u = zs.Normal('u', mu_u, logstd=log_std_u,
                      n_samples=n_particles, group_event_ndims=1)
        mu_v = tf.zeros(shape=[m, D])
        log_std_v = tf.ones(shape=[m, D]) * tf.log(alpha_v)
        v = zs.Normal('v', mu_v, logstd=log_std_v,
                      n_samples=n_particles, group_event_ndims=1)
        gather_u = select_from_axis1(u, select_u)  # [K, batch, D]
        gather_v = select_from_axis1(v, select_v)  # [K, batch, D]
        pred_mu = tf.reduce_sum(gather_u * gather_v, axis=2)
        r = zs.Normal('r', pred_mu, logstd=tf.log(alpha_pred))
    return model, pred_mu


def q_net(observed, n, D, m, n_particles):
    with zs.BayesianNet(observed=observed) as variational:
        mu_v = \
            tf.get_variable('q_mu_v', shape=[m, D],
                            initializer=tf.random_normal_initializer(0, 0.1))
        log_std_v = \
            tf.get_variable('q_log_std_v', shape=[m, 1],
                            initializer=tf.random_normal_initializer(0, 0.1))
        log_std_v = tf.tile(log_std_v, [1, D])
        mu_u = \
            tf.get_variable('q_mu_u', shape=[n, D],
                            initializer=tf.random_normal_initializer(0, 0.1))
        log_std_u = \
            tf.get_variable('q_log_std_u', shape=[n, 1],
                            initializer=tf.random_normal_initializer(0, 0.1))
        log_std_u = tf.tile(log_std_u, [1, D])
        v = zs.Normal('v', mu_v, logstd=log_std_v,
                      n_samples=n_particles, group_event_ndims=1)  # [K, m, D]
        u = zs.Normal('u', mu_u, logstd=log_std_u,
                      n_samples=n_particles, group_event_ndims=1)  # [K, n, D]
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
    N_valid = np.shape(valid_data)[0]
    D = 60
    batch_size = 10000
    test_batch_size = 10000
    valid_batch_size = 10000
    lambda_U = 0.002
    lambda_V = 0.002
    learning_rate = 0.01
    K = 50
    num_steps = 500
    iters = (N_train + batch_size - 1) // batch_size
    test_iters = (N_test + test_batch_size - 1) // test_batch_size
    valid_iters = (N_valid + valid_batch_size - 1) // valid_batch_size
    valid_freq = 10
    test_freq = 10
    result_path = 'tmp/pmf_mean_field/'

    select_u = tf.placeholder(tf.int32, shape=[None, ], name='s_u')
    select_v = tf.placeholder(tf.int32, shape=[None, ], name='s_v')
    n_particles = tf.placeholder(tf.int32, shape=[], name='np')
    alpha_u = 1.0
    alpha_v = 1.0
    alpha_pred = 0.2

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
    true_rating_2d = tf.tile(tf.expand_dims(true_rating, 0), [n_particles, 1])

    # Define models for VI
    def log_joint(observed):
        model, _ = pmf(observed, N, M, D, n_particles, select_u,
                       select_v, alpha_u, alpha_v, alpha_pred)
        log_pu, log_pv = model.local_log_prob(['u', 'v'])  # [K, N], [K, M]
        log_pr = model.local_log_prob('r')  # [K, batch]
        log_pu = tf.reduce_sum(log_pu, axis=1)
        log_pv = tf.reduce_sum(log_pv, axis=1)
        log_pr = tf.reduce_sum(log_pr, axis=1)
        log_pr = log_pr * N_train / tf.cast(tf.shape(select_u)[0],
                                            dtype=tf.float32)
        return log_pu + log_pv + log_pr

    variational = q_net({}, N, D, M, n_particles)
    qv_samples, log_qv = variational.query('v', outputs=True,
                                           local_log_prob=True)
    qu_samples, log_qu = variational.query('u', outputs=True,
                                           local_log_prob=True)
    log_qv = tf.reduce_sum(log_qv, axis=1)
    log_qu = tf.reduce_sum(log_qu, axis=1)
    lower_bound = tf.reduce_mean(
        zs.sgvb(log_joint, {'r': true_rating_2d},
                {'u': [qu_samples, log_qu], 'v': [qv_samples, log_qv]},
                axis=0))
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    _, pred_rating = pmf({'u': qu_samples, 'v': qv_samples,
                          'r': true_rating_2d}, N, M, D, K,
                         select_u, select_v, alpha_u, alpha_v, alpha_pred)
    pred_rating = tf.reduce_mean(pred_rating, axis=0)
    error = pred_rating - true_rating
    rmse = tf.sqrt(tf.reduce_mean(error * error))

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    saver = tf.train.Saver(max_to_keep=10)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, num_steps + 1):
            time_epoch = -time.time()
            res = []
            np.random.shuffle(train_data)
            for t in range(iters):
                ed_pos = min((t + 1) * batch_size, N_train + 1)
                su = train_data[t * batch_size:ed_pos, 0]
                sv = train_data[t * batch_size:ed_pos, 1]
                tr = train_data[t * batch_size:ed_pos, 2]
                _, re = sess.run([infer, rmse, ],
                                 feed_dict={select_u: su,
                                            select_v: sv,
                                            true_rating: tr,
                                            learning_rate_ph: learning_rate,
                                            n_particles: K})
                res.append(re)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): train rmse = {}'.format(
                epoch, time_epoch, np.mean(res)))

            if epoch % valid_freq == 0:
                valid_rmse = []
                valid_rrmse = []
                time_valid = -time.time()
                for t in range(valid_iters):
                    ed_pos = min((t + 1) * valid_batch_size, N_valid + 1)
                    su = valid_data[t * valid_batch_size:ed_pos, 0]
                    sv = valid_data[t * valid_batch_size:ed_pos, 1]
                    tr = valid_data[t * valid_batch_size:ed_pos, 2]
                    re = sess.run(rmse,
                                  feed_dict={select_u: su, select_v: sv,
                                             true_rating: tr,
                                             n_particles: K * 5})
                    valid_rmse.append(re)
                time_valid += time.time()
                print('>>> VALIDATION ({:.1f}s)'.format(time_valid))
                print('>> Validation rmse = {}'.
                      format(np.mean(valid_rmse)))

            if epoch % test_freq == 0:
                test_rmse = []
                test_rrmse = []
                time_test = -time.time()
                for t in range(test_iters):
                    ed_pos = min((t + 1) * test_batch_size, N_test + 1)
                    su = test_data[t * test_batch_size:ed_pos, 0]
                    sv = test_data[t * test_batch_size:ed_pos, 1]
                    tr = test_data[t * test_batch_size:ed_pos, 2]
                    re = sess.run(rmse,
                                  feed_dict={select_u: su, select_v: sv,
                                             true_rating: tr,
                                             n_particles: K * 5})

                    test_rmse.append(re)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test rmse = {}'.
                      format(np.mean(test_rmse)))
