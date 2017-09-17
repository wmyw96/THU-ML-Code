#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
import tensorflow as tf
import numpy as np
from dataset import load_movielens1m_mapped


def random_weights(R, C):
    return tf.random_normal(shape=(R, C), mean=0,
                            stddev=1.0, dtype=tf.float32)


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
    epoches = 300
    batch_size = 100000
    valid_batch_size = 100000
    test_batch_size = 100000
    lambda_U = 0.04
    lambda_V = 0.04
    learning_rate = 0.005
    save_freq = 50
    valid_freq = 10
    test_freq = 10
    iters = (N_train + batch_size - 1) // batch_size
    valid_iters = (N_valid + valid_batch_size - 1) // valid_batch_size
    test_iters = (N_test + test_batch_size - 1) // test_batch_size
    result_path = 'tmp/pmf_map/'

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

    U = tf.Variable(random_weights(N, D))     # (N, D)
    V = tf.Variable(random_weights(M, D))     # (M, D)

    pair_U = tf.placeholder(tf.int32, shape=[None, ], name='s_U')
    pair_V = tf.placeholder(tf.int32, shape=[None, ], name='s_V')
    true_rating = tf.placeholder(tf.float32, shape=[None, ],
                                 name='true_rating')
    num_films = tf.cast(tf.shape(pair_U)[0], tf.float32)

    optimized_U = tf.gather(U, pair_U)  # (batch_size, D)
    optimized_V = tf.gather(V, pair_V)  # (batch_size, D)
    mf_pred_rating = tf.reduce_sum(optimized_U * optimized_V, axis=1)
    constant_rating = tf.convert_to_tensor([0.0], dtype=tf.float32)
    constant_rating = tf.tile(constant_rating, tf.shape(mf_pred_rating))

    exists_u = tf.gather(trained_user, pair_U)
    exists_v = tf.gather(trained_movie, pair_V)
    exists_uv = tf.logical_and(exists_u, exists_v)
    pred_rating = tf.where(exists_uv, mf_pred_rating, constant_rating)

    pred_rating = tf.sigmoid(pred_rating)
    trans_rating = (true_rating - 1.0) / 4.0

    error = pred_rating - trans_rating
    old_error = tf.sigmoid(mf_pred_rating) - trans_rating
    rmse = tf.sqrt(tf.reduce_mean(error * error)) * 4
    old_rmse = tf.sqrt(tf.reduce_mean(old_error * old_error)) * 4
    cost = 0.5 * tf.reduce_sum(error * error) * 4 * \
        ((N_train + 0.0) / num_films) + \
        lambda_U * 0.5 * tf.reduce_sum(U * U, axis=[0, 1]) + \
        lambda_V * 0.5 * tf.reduce_sum(V * V, axis=[0, 1])

    grads = optimizer.compute_gradients(cost)
    infer = optimizer.apply_gradients(grads)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    saver = tf.train.Saver(max_to_keep=10)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(result_path)
        begin_epoch = 1
        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

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
                    re, ore = sess.run([rmse, old_rmse],
                                       feed_dict={pair_U: su, pair_V: sv,
                                                  true_rating: tr})
                    valid_rmse.append(re)
                    valid_rrmse.append(ore)
                time_valid += time.time()
                print('>>> VALIDATION ({:.1f}s)'.format(time_valid))
                print('>> Validation rmse = {}, uncorrected rmse = {}'.
                      format(np.mean(valid_rmse), np.mean(valid_rrmse)))

            if epoch % test_freq == 0:
                test_rmse = []
                test_rrmse = []
                time_test = -time.time()
                for t in range(test_iters):
                    ed_pos = min((t + 1) * test_batch_size, N_test + 1)
                    su = test_data[t * test_batch_size:ed_pos, 0]
                    sv = test_data[t * test_batch_size:ed_pos, 1]
                    tr = test_data[t * test_batch_size:ed_pos, 2]
                    re, ore = sess.run([rmse, old_rmse],
                                       feed_dict={pair_U: su, pair_V: sv,
                                                  true_rating: tr})
                    test_rmse.append(re)
                    test_rrmse.append(ore)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test rmse = {}, uncorrected rmse = {}'.
                      format(np.mean(test_rmse), np.mean(test_rrmse)))

            if epoch % save_freq == 0:
                save_path = os.path.join(result_path,
                                         "pmf_map.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                saver.save(sess, save_path)
