#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
import tensorflow as tf
import numpy as np
from dataset import load_movielens1m_mapped

if __name__ == '__main__':
    np.random.seed(1234)
    tf.set_random_seed(1237)
    M, N, train_data, valid_data, test_data, nla, nlb, nlc, nld = \
        load_movielens1m_mapped('data/ml-1m.zip')

    # set configurations and hyper parameters
    N_train = np.shape(train_data)[0]
    N_valid = np.shape(valid_data)[0]
    N_test = np.shape(test_data)[0]
    n_hidden = 15
    n_anchors = 50
    D = 30
    epoches = 100
    batch_size = 100000
    valid_batch_size = 100000
    test_batch_size = 100000
    lambda_u = 0.002
    lambda_v = 0.002
    learning_rate = 0.01
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

    h_u = 0.8
    h_v = 0.8
    prior_u = tf.get_variable('U', shape=[N, D],
                              initializer=tf.random_normal_initializer(0, .01))
    prior_v = tf.get_variable('V', shape=[M, D],
                              initializer=tf.random_normal_initializer(0, .01))
    anchors_uid = np.random.randint(N, size=n_anchors)
    anchors_vid = np.random.randint(M, size=n_anchors)
    anchors_uid = tf.constant(anchors_uid, dtype=tf.int32)
    anchors_vid = tf.constant(anchors_vid, dtype=tf.int32)
    anchors_u = tf.gather(prior_u, anchors_uid)   # [n_a, D]
    anchors_v = tf.gather(prior_v, anchors_vid)   # [n_b, D]
    anchors_u_tr = tf.transpose(anchors_u)
    anchors_v_tr = tf.transpose(anchors_v)
    dist_u = tf.matmul(prior_u, anchors_u_tr)
    dist_u = dist_u / tf.sqrt(tf.reduce_sum(prior_u * prior_u)) / \
        tf.sqrt(tf.reduce_sum(anchors_u_tr * anchors_u_tr))
    dist_v = tf.matmul(prior_v, anchors_v_tr)
    dist_v = dist_v / tf.sqrt(tf.reduce_sum(prior_v * prior_v)) / \
        tf.sqrt(tf.reduce_sum(anchors_v_tr * anchors_v_tr))
    kernel_u = tf.maximum(1 - dist_u * dist_u / h_u / h_u, 0)
    kernel_v = tf.maximum(1 - dist_v * dist_v / h_u / h_u, 0)

    u = tf.get_variable('u', shape=[N, n_anchors, n_hidden],
                        initializer=tf.random_normal_initializer(0, 0.01))
    v = tf.get_variable('v', shape=[M, n_anchors, n_hidden],
                        initializer=tf.random_normal_initializer(0, 0.01))

    select_idxu = tf.placeholder(tf.int32, shape=[None, ], name='s_u')
    select_idxv = tf.placeholder(tf.int32, shape=[None, ], name='s_v')
    true_rating = tf.placeholder(tf.float32, shape=[None, ],
                                 name='true_rating')
    num_films = tf.cast(tf.shape(select_idxu)[0], tf.float32)

    select_u = tf.gather(u, select_idxu)  # (batch_size, n_a, n_h)
    select_v = tf.gather(v, select_idxv)  # (batch_size, n_a, n_h)
    select_kernel_u = tf.gather(kernel_u, select_idxu)
    select_kernel_v = tf.gather(kernel_v, select_idxv)
    select_kernel = select_kernel_u * select_kernel_v
    sp_pred_rating = tf.reduce_sum(select_u * select_v, axis=2)  # (bs, n_a)
    error = sp_pred_rating - tf.expand_dims(true_rating, 1)
    cost = tf.reduce_sum(error * error * select_kernel)
    cost = cost * N_train / tf.cast(num_films, tf.float32) + \
        lambda_u * tf.reduce_sum(u * u) + lambda_v * tf.reduce_sum(v * v)
    pred = select_kernel / tf.reduce_sum(select_kernel, axis=1, keep_dims=True)
    pred_rating = tf.reduce_sum(pred * sp_pred_rating, axis=1)

    error = pred_rating - true_rating
    rmse = tf.sqrt(tf.reduce_mean(error * error))

    grads = optimizer.compute_gradients(cost)
    xgrads = [grads[2], grads[3]]
    infer = optimizer.apply_gradients(xgrads)

    select_u_pre = tf.gather(prior_u, select_idxu)
    select_v_pre = tf.gather(prior_v, select_idxv)
    error_pre = tf.reduce_sum(select_u_pre * select_v_pre, axis=1)
    error_pre = error_pre - true_rating
    rmse_pre = tf.sqrt(tf.reduce_mean(error_pre * error_pre))
    cost_pre = tf.reduce_mean(error_pre * error_pre) * N_train + 0.01 * \
        (tf.reduce_sum(prior_u * prior_u) + tf.reduce_sum(prior_v * prior_v))
    optimizer_pre = tf.train.AdamOptimizer(learning_rate_ph, beta1=0.5)
    grads_pre = optimizer.compute_gradients(cost_pre)
    infer_pre = optimizer.apply_gradients(grads_pre)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Restore from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(result_path)
        begin_epoch = 1
        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        for epoch in range(begin_epoch, 50 + 1):
            time_epoch = -time.time()
            res = []
            np.random.shuffle(train_data)
            for t in range(iters):
                ed_pos = min((t + 1) * batch_size, N_train + 1)
                su = train_data[t * batch_size:ed_pos, 0]
                sv = train_data[t * batch_size:ed_pos, 1]
                tr = train_data[t * batch_size:ed_pos, 2]
                _, re = sess.run([infer_pre, rmse_pre],
                                 feed_dict={select_idxu: su,
                                            select_idxv: sv,
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
                    re = sess.run(rmse_pre, feed_dict={select_idxu: su,
                                                       select_idxv: sv,
                                                       true_rating: tr})
                    valid_rmse.append(re)
                    valid_rrmse.append(re)
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
                    re = sess.run(rmse_pre, feed_dict={select_idxu: su,
                                                       select_idxv: sv,
                                                       true_rating: tr})
                    test_rmse.append(re)
                    test_rrmse.append(re)
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

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            res = []
            np.random.shuffle(train_data)
            for t in range(iters):
                ed_pos = min((t + 1) * batch_size, N_train + 1)
                su = train_data[t * batch_size:ed_pos, 0]
                sv = train_data[t * batch_size:ed_pos, 1]
                tr = train_data[t * batch_size:ed_pos, 2]
                _, re = sess.run([infer, rmse, ],
                                 feed_dict={select_idxu: su,
                                            select_idxv: sv,
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
                    re = sess.run(rmse, feed_dict={select_idxu: su,
                                                   select_idxv: sv,
                                                   true_rating: tr})
                    valid_rmse.append(re)
                    valid_rrmse.append(re)
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
                    re = sess.run(rmse, feed_dict={select_idxu: su,
                                                   select_idxv: sv,
                                                   true_rating: tr})
                    test_rmse.append(re)
                    test_rrmse.append(re)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test rmse = {}, uncorrected rmse = {}'.
                      format(np.mean(test_rmse), np.mean(test_rrmse)))

