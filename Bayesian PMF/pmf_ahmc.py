#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
        u = zs.Normal('u', mu_u, log_std_u,
                      n_samples=n_particles, group_event_ndims=1)
        mu_v = tf.zeros(shape=[m, D])
        log_std_v = tf.ones(shape=[m, D]) * tf.log(alpha_v)
        v = zs.Normal('v', mu_v, log_std_v,
                      n_samples=n_particles, group_event_ndims=1)
        gather_u = select_from_axis1(u, select_u)  # [K, batch, D]
        gather_v = select_from_axis1(v, select_v)  # [K, batch, D]
        pred_mu = tf.reduce_sum(gather_u * gather_v, axis=2)
        r = zs.Normal('r', pred_mu, tf.log(alpha_pred))
    return model, pred_mu


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
    batch_size = 100
    test_batch_size = 100000
    lambda_U = 0.002
    lambda_V = 0.002
    learning_rate = 0.005
    K = 8
    num_steps = 500
    iters = (N_train + batch_size - 1) // batch_size
    test_iters = (N_test + test_batch_size - 1) // test_batch_size
    result_path = 'tmp/pmf_ahmc/'

    select_u = tf.placeholder(tf.int32, shape=[None, ], name='s_u')
    select_v = tf.placeholder(tf.int32, shape=[None, ], name='s_v')
    subselect_u = tf.placeholder(tf.int32, shape=[None, ], name='ss_u')
    subselect_v = tf.placeholder(tf.int32, shape=[None, ], name='ss_v')
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

    # Define samples as variables
    Us = []
    Vs = []
    for i in range(N):
        ui = tf.get_variable('u%d' % i, shape=[K, 1, D],
                             initializer=tf.random_normal_initializer(0, 0.1),
                             trainable=False)
        Us.append(ui)
    for i in range(M):
        vi = tf.get_variable('v%d' % i, shape=[K, 1, D],
                             initializer=tf.random_normal_initializer(0, 0.1),
                             trainable=False)
        Vs.append(vi)
    U = tf.concat(Us, axis=1)
    V = tf.concat(Vs, axis=1)

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

    hmc_u = zs.HMC(step_size=1e-3, n_leapfrogs=10, adapt_step_size=None,
                   target_acceptance_rate=0.9)
    hmc_v = zs.HMC(step_size=1e-3, n_leapfrogs=10, adapt_step_size=None,
                   target_acceptance_rate=0.9)
    target_u = select_from_axis1(U, select_u)
    target_v = select_from_axis1(V, select_v)

    candidate_sample_u = \
        tf.get_variable('cand_sample_u', shape=[K, 1, D],
                        initializer=tf.random_normal_initializer(0, 0.1),
                        trainable=True)
    candidate_sample_v = \
        tf.get_variable('cand_sample_v', shape=[K, 1, D],
                        initializer=tf.random_normal_initializer(0, 0.1),
                        trainable=True)
    sample_u_op, sample_u_info = \
        hmc_u.sample(log_joint, {'r': true_rating_2d, 'v': target_v},
                     {'u': candidate_sample_u})
    sample_v_op, sample_v_info = \
        hmc_v.sample(log_joint, {'r': true_rating_2d, 'u': target_u},
                     {'v': candidate_sample_v})

    candidate_idx_u = tf.placeholder(tf.int32, shape=[1, ], name='cand_ui')
    candidate_idx_v = tf.placeholder(tf.int32, shape=[1, ], name='cand_vi')
    candidate_u = select_from_axis1(U, candidate_idx_u)
    candidate_v = select_from_axis1(V, candidate_idx_v)

    trans_cand_U = tf.assign(candidate_sample_u, candidate_u)
    trans_cand_V = tf.assign(candidate_sample_v, candidate_v)

    trans_us_cand = []
    for i in range(N):
        trans_us_cand.append(tf.assign(Us[i], candidate_sample_u))
    trans_vs_cand = []
    for i in range(M):
        trans_vs_cand.append(tf.assign(Vs[i], candidate_sample_v))

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
