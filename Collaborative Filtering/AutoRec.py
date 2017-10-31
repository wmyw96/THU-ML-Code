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
                user_movie_score[user_id][j]
            cc += 1
        mask.append(cur_mask)
        rating.append(cur_rating)
    mask_batch = np.array(mask)
    rating_batch = np.array(rating)  # enforce var to be 1
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
                user_movie_score[user_id][j]
        mask.append(cur_mask)
        rating.append(cur_rating)

        cur_bmask = [0] * M
        cur_brating = [0] * M
        for j in range(len(user_movie_test[user_id])):
            cur_bmask[user_movie_test[user_id][j]] = 1
            cur_brating[user_movie_test[user_id][j]] = \
                user_movie_score_test[user_id][j]
        bmask.append(cur_bmask)
        brating.append(cur_brating)

    infer_mask = np.array(mask)
    infer_rating = np.array(rating)  # enforce var to be 1
    mask_batch = np.array(bmask)
    rating_batch = np.array(brating)
    return infer_mask, infer_rating, mask_batch, rating_batch


if __name__ == '__main__':
    np.random.seed(1234)
    tf.set_random_seed(1237)
    M, N, train_data, valid_data, test_data, user_movie, \
        user_movie_score, movie_user, movie_user_score, \
        user_movie_test, user_movie_score_test,\
        user_movie_valid, user_movie_score_valid, \
        = load_movielens1m_mapped_ptest('data/ml-1m.zip', valid_map=True)

    print(N)
    print(M)
    # set configurations and hyper parameters
    N_train = np.shape(train_data)[0]
    N_test = np.shape(test_data)[0]
    N_valid = np.shape(valid_data)[0]
    n_hidden = 50
    batch_size = 1000
    test_batch_size = 1000
    valid_batch_size = 1000
    num_epochs = 1000
    learning_rate = 1e-3 * 5
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    iters = (N + batch_size - 1) // batch_size
    test_iters = (N + test_batch_size - 1) // test_batch_size
    valid_iters = (N + valid_batch_size - 1) // valid_batch_size
    result_path = 'tmp/autorec/'

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

    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    infer_mask = tf.placeholder(tf.float32, shape=[None, M], name='infer_mask')
    infer_rating = tf.placeholder(tf.float32, shape=[None, M],
                                  name='infer_rating')
    keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
    gen_mask = tf.placeholder(tf.float32, shape=[None, M], name='gen_mask')
    gen_rating = tf.placeholder(tf.float32, shape=[None, M], name='gen_rating')
    n = tf.shape(gen_mask)[0]

    V = tf.get_variable('v', shape=[M, n_hidden],
                        initializer=tf.random_normal_initializer(0, 2 / np.sqrt(M + n_hidden)))
    b = tf.get_variable('b', shape=[1, n_hidden],
                        initializer=tf.constant_initializer(0))
    # [n, hidden]
    input_mask = tf.identity(infer_mask)
    output_mask = tf.identity(gen_mask)
    input_layer = input_mask * infer_rating

    hidden_a = tf.matmul(input_layer, V) + b
    hidden = tf.sigmoid(hidden_a)

    right_term = tf.reduce_sum(V * V, axis=0, keep_dims=True)
    left_term = tf.square(hidden * (1 - hidden))
    contractive_penaty = tf.reduce_sum(left_term * right_term)

    U = tf.get_variable('u', shape=[n_hidden, M],
                        initializer=tf.random_normal_initializer(0, 2 / np.sqrt(M + n_hidden)))
    mu = tf.get_variable('mu', shape=[1, M],
                         initializer=tf.constant_initializer(0))
    pred = tf.matmul(hidden, U) + mu  # [n, M]
    error = (pred - gen_rating) * output_mask
    se = tf.reduce_sum(error * error)
    lbda = 0
    lbda2 = 50
    cost = N_train / tf.cast(tf.reduce_sum(output_mask), tf.float32) * \
        tf.reduce_sum(error * error) + \
        lbda * (tf.reduce_sum(U * U) + tf.reduce_sum(V * V))
    # Test for Contractive_penaty
    #hiddens = tf.split(hidden, [1] * n_hidden, axis=1)
    #penaties = []
    #for i in range(n_hidden):
    #    contrast_grad = tf.gradients(hiddens[i], input_layer)[0]
    #    g2 = tf.reduce_sum(contrast_grad * contrast_grad, axis=0)
    #    penaties.append(tf.reduce_sum(g2))

    # Test for Contractive Penaty 2
    #preds = tf.split(pred, [1] * M, axis=1)
    #penaties = []
    #for i in range(M):
    #    contract_grad = tf.gradients(preds[i], input_layer)[0]
    #    g2 = tf.reduce_sum(contract_grad * contract_grad, axis=0)
    #    mask_can = [1] * M
    #    mask_can[i] = 0
    #    mask = tf.constant(mask_can, dtype=tf.float32)
    #    penaties.append(tf.reduce_sum(g2 * mask))

    right_term2 = tf.tile(tf.expand_dims(U, axis=0), [n, 1, 1])
    left_term2 = tf.expand_dims(hidden * (1 - hidden), axis=1) * \
        tf.expand_dims(V, axis=0)
    #print(left_term2.get_shape())
    #print(right_term2.get_shape())
    prod = tf.matmul(left_term2, right_term2)
    mask = tf.expand_dims(tf.ones((M, M)) - tf.eye(M), 0)
    contractive_penaty2 = tf.reduce_sum(prod * prod)

    #diff = sum(penaties) - contractive_penaty2
    #ccc = sum(penaties)
    cost = cost + contractive_penaty2 * lbda2

    number = tf.reduce_sum(output_mask)

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph)    # Define models

    grads = optimizer.compute_gradients(cost)
    capped_gvs = \
        [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grads]
    infer = optimizer.apply_gradients(capped_gvs)

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
                                 feed_dict={is_training: True,
                                            infer_mask: tr_mask,
                                            infer_rating: tr_rating,
                                            gen_mask: tr_mask,
                                            gen_rating: tr_rating,
                                            learning_rate_ph: learning_rate})
                ses.append(__)
            epoch_time += time.time()
            print('Epoch {}({:.1f}s): rmse = {}'.format(
                epoch + 1, epoch_time, np.sqrt(np.sum(ses) / N_train)))

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
                                  feed_dict={is_training: False,
                                             infer_mask: in_mask,
                                             infer_rating: in_rating,
                                             gen_mask: out_mask,
                                             gen_rating: out_rating,
                                             learning_rate_ph: learning_rate})
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
                    in_mask, in_rating, out_mask, out_rating = \
                        get_test_data(M, l, r, user_movie, user_movie_score,
                                      user_movie_test, user_movie_score_test)
                    __ = sess.run(se,
                                  feed_dict={is_training: False,
                                             infer_mask: in_mask,
                                             infer_rating: in_rating,
                                             gen_mask: out_mask,
                                             gen_rating: out_rating,
                                             learning_rate_ph: learning_rate})
                    test_se.append(__)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test rmse = {}'.format(
                    (np.sqrt(np.sum(test_se) / N_test))))
