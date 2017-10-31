#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
import tensorflow as tf
import numpy as np
from dataset import load_movielens1m_mapped


def random_weights(t):
    return tf.random_normal(shape=t, mean=0,
                            stddev=0.1, dtype=tf.float32)


if __name__ == '__main__':
    np.random.seed(1234)
    tf.set_random_seed(1237)
    M, N, train_data, valid_data, test_data, nla, nlb, nlc, nld = \
        load_movielens1m_mapped('data/ml-1m.zip')

    # set configurations and hyper parameters
    N_train = np.shape(train_data)[0]
    N_valid = np.shape(valid_data)[0]
    N_test = np.shape(test_data)[0]
    D = 30
    D2 = 500
    epoches = 1000
    batch_size = 100000
    valid_batch_size = 100000
    test_batch_size = 100000
    lambda_U = 0.1
    lambda_V = 0.1
    learning_rate = 0.005
    save_freq = 50
    valid_freq = 10
    test_freq = 10
    iters = (N_train + batch_size - 1) // batch_size
    valid_iters = (N_valid + valid_batch_size - 1) // valid_batch_size
    test_iters = (N_test + test_batch_size - 1) // test_batch_size

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

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, beta1=0.5)

    U = tf.Variable(random_weights((N, D)))     # (N, D)
    vs = []
    regularizer = 0.0
    pre = 0.0

    for i in range(5):
        cur = tf.Variable(random_weights((M, 1, D)))
        regularizer += tf.reduce_sum(cur * cur)
        cur = cur + pre
        vs.append(cur)
        pre = cur

    V = tf.concat(vs, axis=1)    # (M, D)

    pair_U = tf.placeholder(tf.int32, shape=[None, ], name='s_U')
    pair_V = tf.placeholder(tf.int32, shape=[None, ], name='s_V')
    true_rating = tf.placeholder(tf.int32, shape=[None, ],
                                 name='true_rating')
    num_films = tf.cast(tf.shape(pair_U)[0], tf.float32)

    optimized_U = tf.gather(U, pair_U)  # (batch_size, D)
    optimized_V = tf.gather(V, pair_V)  # (batch_size, 5, D)
    optimized_U = tf.tile(tf.expand_dims(optimized_U, 1), [1, 5, 1])
    mf_pred_score = tf.reduce_sum(optimized_U * optimized_V, axis=2)
    mf_pred_prob = tf.nn.softmax(mf_pred_score, dim=-1)
    onehot_rating = tf.one_hot(true_rating - 1, depth=5)
    cross_entropy = tf.reduce_sum(-tf.log(mf_pred_prob + 1e-8) * onehot_rating, axis=1)
    
    weight = tf.expand_dims(tf.constant([1, 2, 3, 4, 5], tf.float32), 0)
    pred_rating = tf.reduce_sum(mf_pred_prob * weight, axis=1)
    error = pred_rating - tf.cast(true_rating, tf.float32)
    rmse = tf.sqrt(tf.reduce_mean(error * error))
    cost = 0.5 * tf.reduce_mean(cross_entropy) * (N_train + 0.0) + \
        lambda_U * 0.5 * tf.reduce_sum(U * U, axis=[0, 1]) + \
        lambda_V * 0.5 * regularizer

    grads = optimizer.compute_gradients(cost)
    infer = optimizer.apply_gradients(grads)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        begin_epoch = 1

        for epoch in range(begin_epoch, epoches + 1):
            time_epoch = -time.time()
            res = []
            np.random.shuffle(train_data)
            for t in range(iters):
                ed_pos = min((t + 1) * batch_size, N_train + 1)
                su = train_data[t * batch_size:ed_pos, 0]
                sv = train_data[t * batch_size:ed_pos, 1]
                tr = train_data[t * batch_size:ed_pos, 2]
                _, re = sess.run([infer, rmse, ],
                                 feed_dict={pair_U: su,
                                            pair_V: sv,
                                            true_rating: tr,
                                            learning_rate_ph: learning_rate})
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
                                       feed_dict={pair_U: su, pair_V: sv,
                                                  true_rating: tr})
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
                                       feed_dict={pair_U: su, pair_V: sv,
                                                  true_rating: tr})
                    test_rmse.append(re)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test rmse = {}'.
                      format(np.mean(test_rmse)))

