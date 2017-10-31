#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import numpy as np
import zhusuan as zs
from dataset import load_movielens1m_mapped_ptest
from tensorflow.contrib import layers
import random
import sys


def select_from_axis(para, indices, axis, rank):
    perm = range(rank)
    perm[axis] = 0
    perm[0] = axis
    gather_para = tf.transpose(para, perm=perm)
    gather_para = tf.gather(gather_para, indices)
    gather_para = tf.transpose(gather_para, perm=perm)
    return gather_para


def pmf(observed, n, m, num_ratings, D, n_particles, select_uid, select_vid,
        alpha_u, alpha_v, alpha_pred, is_training):
    with zs.BayesianNet(observed=observed) as model:
        mu_z = tf.zeros(shape=[n, D])
        log_std_z = tf.ones(shape=[n, D]) * tf.log(alpha_u)
        z = zs.Normal('z', mu_z, logstd=log_std_z,
                      n_samples=n_particles, group_event_ndims=1)  # [K, n, D]
        mu_v = tf.zeros(shape=[m, num_ratings, D])
        log_std_v = tf.ones(shape=[m, num_ratings, D]) * tf.log(alpha_v)
        v = zs.Normal('v', mu_v, logstd=log_std_v,
                      n_samples=n_particles, group_event_ndims=1)  # [K, m, nR, D]
        select_u = select_from_axis(z, select_uid, 1, 3)   # [K, B, D]
        select_v = select_from_axis(v, select_vid, 1, 4)   # [K, B, nR, D]
        select_u = tf.tile(tf.expand_dims(select_u, 2), [1, 1, num_ratings, 1])
        pred_score = tf.reduce_sum(select_u * select_v, axis=3)
        const_rating = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32) # [nR]
        const_rating = tf.expand_dims(tf.expand_dims(const_rating, 0), 0)
        pred_mu = \
            tf.reduce_sum(tf.nn.softmax(pred_score, dim=2) * const_rating, axis=2)
        print(pred_score.get_shape())
        r = zs.OnehotCategorical('r', logits=pred_score)   # [K, B]
    return model, pred_mu, tf.nn.softmax(pred_score, dim=2)


def get_accu(K):
    x = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1):
            x[i, j] = 1
    return tf.constant(x, dtype=tf.float32)


def q_net(observed, ratings, indices, portion, n, D, m, n_ratings, n_particles,
          is_training, kp_dropout):
    with zs.BayesianNet(observed=observed) as variational:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        mu_v = \
            tf.get_variable('q_mu_v', shape=[m, n_ratings, D],
                            initializer=tf.random_normal_initializer(0, 0.3))
        accu = get_accu(n_ratings)
        accu = tf.tile(tf.expand_dims(accu, 0), [m, 1, 1])
        mu_v = tf.matmul(accu, mu_v)
        log_std_v = \
            tf.get_variable('q_log_std_v', shape=[m, n_ratings, D],
                            initializer=tf.random_normal_initializer(0, 0.01))
        log_std_v = tf.matmul(accu, log_std_v)
        #log_std_v = tf.tile(log_std_v, [1, 1, D])
        v = zs.Normal('v', mu_v, logstd=log_std_v,
                      n_samples=n_particles, group_event_ndims=1)  # [K, m, nR, D]
        input_v = select_from_axis(v, indices, 1, 4)       # [K, np, nR, D]
        one_hot_rt = tf.one_hot(ratings - 1, depth=5)
        one_hot_rt = tf.expand_dims(tf.expand_dims(one_hot_rt, 0), 3)
        input_v = tf.reduce_sum(one_hot_rt * input_v, axis=2)
        input_mask = tf.nn.dropout(tf.ones(shape=[n_particles,
                                                  tf.shape(indices)[0]]),
                                   keep_prob=kp_dropout)
        input_mask = tf.tile(tf.expand_dims(input_mask, 2), [1, 1, D])
        input_i = input_v * input_mask  # [K, np, D+1]
        lh_r = layers.fully_connected(input_i, 200)
        lh_r = layers.fully_connected(lh_r, 200)
        hd = \
            tf.matmul(
                tf.tile(tf.expand_dims(portion, 0), [n_particles, 1, 1]), lh_r)
        lz_h = layers.fully_connected(hd, 200)
        z_mean = layers.fully_connected(lz_h, D, activation_fn=None)
        z_log_std = layers.fully_connected(lz_h, D, activation_fn=None)
        z = zs.Normal('z', z_mean, logstd=z_log_std, n_samples=None,
                      group_event_ndims=1)
    return variational


def get_traing_data(M, head, tail, idx_list, user_movie, user_movie_score):
    i_indices = []
    i_ratings = []
    i_coeff = []
    cc = 0
    uid = []
    vid = []
    for i in range(tail - head):
        user_id = idx_list[head + i]

        uid += [i] * len(user_movie[user_id])
        vid += user_movie[user_id]
        i_ratings += user_movie_score[user_id]
        cc += len(user_movie[user_id])

        i_indices = i_indices + user_movie[user_id]
        i_coeff = i_coeff + [len(user_movie[user_id])]

    i_portion = np.zeros((len(i_coeff), sum(i_coeff)))
    id = 0
    for i in range(len(i_coeff)):
        for j in range(i_coeff[i]):
            i_portion[i][id] = 1.0 / i_coeff[i]
            id += 1
    return i_indices, i_ratings, i_portion, uid, vid, i_ratings, cc


def get_test_data(M, head, tail, user_movie, user_movie_score,
                  user_movie_test, user_movie_score_test):
    i_indices = []
    i_ratings = []
    i_coeff = []
    uid = []
    vid = []
    ratings = []
    for i in range(tail - head):
        user_id = head + i

        i_ratings += user_movie_score[user_id]
        i_coeff += [len(user_movie_score[user_id])]
        i_indices += user_movie[user_id]

        uid += [i] * len(user_movie_test[user_id])
        vid += user_movie_test[user_id]
        ratings += user_movie_score_test[user_id]

    i_portion = np.zeros((len(i_coeff), sum(i_coeff)))
    id = 0
    for i in range(len(i_coeff)):
        for j in range(i_coeff[i]):
            i_portion[i][id] = 1.0 / i_coeff[i]
            id += 1
    return i_indices, i_ratings, i_portion, uid, vid, ratings


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
    K = 5
    num_epochs = 1000
    n_rts = 5
    learning_rate = float(sys.argv[1])
    print('learning rate = {}'.format(learning_rate))
    anneal_lr_freq = 100
    anneal_lr_rate = 0.75
    iters = (N + batch_size - 1) // batch_size
    test_iters = (N + test_batch_size - 1) // test_batch_size
    valid_iters = (N + valid_batch_size - 1) // valid_batch_size
    result_path = 'tmp/pmf_vi/'

    hp_alpha_u = 1.0
    hp_alpha_v = 1.0
    hp_alpha_pred = float(sys.argv[2])
    print('alpha_pred = {}'.format(hp_alpha_pred))
    hp_alpha_bias = 1.0
    constant_kp = float(sys.argv[3])
    print('keep_prob = {}'.format(constant_kp))
    keep_prob = tf.placeholder(tf.float32, shape=[], name='kp')

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
    infer_indices = \
        tf.placeholder(tf.int32, shape=[None], name='infer_indices')
    infer_ratings = \
        tf.placeholder(tf.int32, shape=[None], name='infer_ratings')
    infer_portion = \
        tf.placeholder(tf.float32, shape=[None, None], name='infer_portion')
    gen_uid = tf.placeholder(tf.int32, shape=[None, ], name='gen_uid')
    gen_vid = tf.placeholder(tf.int32, shape=[None, ], name='gen_vid')
    gen_rating = tf.placeholder(tf.int32, shape=[None, ], name='gen_rating')

    def log_joint(observed):
        model, _, __ = pmf(observed, n, M, n_rts, n_z, n_particles, gen_uid, gen_vid,
                       hp_alpha_u, hp_alpha_v, hp_alpha_pred, is_training)
        log_pz, log_pv, log_pr = \
            model.local_log_prob(['z', 'v', 'r'])
        #print(log_pr.get_shape())
        log_pr = tf.reduce_sum(log_pr, axis=1)
        log_pz = tf.reduce_sum(log_pz, axis=1)
        log_pv = tf.reduce_sum(log_pv, axis=1)
        log_pv =  tf.reduce_sum(log_pv, axis=1)
        #print('ff')
        #print(log_pr.get_shape())
        #print(log_pz.get_shape())
        #print(log_pv.get_shape())
        #print('ef')
        return log_pz + log_pv + log_pr  # [K]

    variational = q_net({}, infer_ratings, infer_indices, infer_portion,
                        n, n_z, M, n_rts, n_particles, is_training, keep_prob)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    qv_samples, log_qv = variational.query('v', outputs=True,
                                           local_log_prob=True)
    log_qz = tf.reduce_sum(log_qz, axis=1)
    log_qv = tf.reduce_sum(log_qv, axis=1)
    log_qv = tf.reduce_sum(log_qv, axis=1)
    #print(log_qz.get_shape())
    #print(log_qv.get_shape())

    onehot_rating = tf.one_hot(gen_rating - 1, depth=5)
    onehot_rating = tf.cast(onehot_rating, tf.int32)
    onehot_obs = tf.tile(tf.expand_dims(onehot_rating, 0), [n_particles, 1, 1])
    lower_bound = tf.reduce_mean(
        zs.sgvb(log_joint, {'r': onehot_obs},
                {'z': [qz_samples, log_qz],
                 'v': [qv_samples, log_qv]}, axis=0))

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    # Prediction and SE calculation
    _, pred, score = pmf({'z': qz_samples, 'v': qv_samples, 'r': gen_rating},
                  n, M, n_rts, n_z, n_particles, gen_uid, gen_vid,
                  hp_alpha_u, hp_alpha_v, hp_alpha_pred, is_training)
    pred = tf.reduce_mean(pred, axis=0)
    prob = tf.reduce_mean(score, axis=0)
    acc = tf.reduce_mean(tf.reduce_sum(prob * onehot_rating, axis=1), axis=0)
    error = (pred - tf.cast(gen_rating, dtype=tf.float32))
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
            accs = []
            idxes = range(N)
            random.shuffle(idxes)
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate

            for t in range(iters):
                l = t * batch_size
                r = min((t + 1) * batch_size, N)
                cur_batch_size = r - l
                tr_ins, tr_rts, tr_port, tr_u, tr_v, tr_rating, cc = \
                    get_traing_data(M, l, r, idxes, user_movie,
                                    user_movie_score)
                _, __, ___, xx = sess.run([infer, se, acc, onehot_rating],
                                 feed_dict={n: cur_batch_size,
                                            m: M,
                                            is_training: True,
                                            n_particles: K,
                                            infer_indices: tr_ins,
                                            infer_ratings: tr_rts,
                                            infer_portion: tr_port,
                                            gen_uid: tr_u,
                                            gen_vid: tr_v,
                                            gen_rating: tr_rating,
                                            learning_rate_ph: learning_rate,
                                            keep_prob: constant_kp})
                ses.append(__)
                accs.append(___)
                print(xx)
            epoch_time += time.time()
            print('Epoch {}({:.1f}s): rmse = {}, acc = {}'.format(
                epoch + 1, epoch_time, np.sqrt(np.sum(ses) / N_train),
                np.mean(accs)))

            if (epoch + 1) % 10 == 0:
                test_se = []
                time_test = -time.time()
                for t in range(valid_iters):
                    l = t * valid_batch_size
                    r = min((t + 1) * valid_batch_size, N)
                    cur_batch_size = r - l
                    in_ins, in_rts, in_port, out_uid, out_vid, out_rating = \
                        get_test_data(M, l, r, user_movie, user_movie_score,
                                      user_movie_valid, user_movie_score_valid)
                    __ = sess.run(se,
                                  feed_dict={n: cur_batch_size,
                                             m: M,
                                             is_training: False,
                                             n_particles: K,
                                             infer_indices: in_ins,
                                             infer_ratings: in_rts,
                                             infer_portion: in_port,
                                             gen_uid: out_uid,
                                             gen_vid: out_vid,
                                             gen_rating: out_rating,
                                             learning_rate_ph: learning_rate,
                                             keep_prob: 1.0})
                    test_se.append(__)
                time_test += time.time()
                print('>>> VALIDATION ({:.1f}s)'.format(time_test))
                print('>> Valid rmse = {}'.format(
                    (np.sqrt(np.sum(test_se) / N_valid))))

            if (epoch + 1) % 10 == 0:
                test_se = []
                time_test = -time.time()
                for t in range(test_iters):
                    l = t * test_batch_size
                    r = min((t + 1) * test_batch_size, N)
                    cur_batch_size = r - l
                    in_ins, in_rts, in_port, out_uid, out_vid, out_rating = \
                        get_test_data(M, l, r, user_movie, user_movie_score,
                                      user_movie_test, user_movie_score_test)
                    __ = sess.run(se,
                                  feed_dict={n: cur_batch_size,
                                             m: M,
                                             is_training: False,
                                             n_particles: K,
                                             infer_indices: in_ins,
                                             infer_ratings: in_rts,
                                             infer_portion: in_port,
                                             gen_uid: out_uid,
                                             gen_vid: out_vid,
                                             gen_rating: out_rating,
                                             learning_rate_ph: learning_rate,
                                             keep_prob: 1.0})
                    test_se.append(__)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test rmse = {}'.format(
                    (np.sqrt(np.sum(test_se) / N_test))))

