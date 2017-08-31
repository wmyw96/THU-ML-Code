#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import time
import os
import re
import tensorflow as tf
import numpy as np
import zhusuan as zs


def random_weights(R, C):
    return tf.random_normal(shape=(R, C), mean=0,
                            stddev=0.001, dtype=tf.float32)


def read_data():
    train_filename = ['data/combined_data_1.txt', 'data/combined_data_2.txt',
                      'data/combined_data_3.txt', 'data/combined_data_4.txt']
    test_filename = 'data/qualifying.txt'

    corpus = []
    num_users = 0
    user_map = {}
    movie_user = {}
    user_movie = {}
    movie_user_score = {}
    user_movie_score = {}
    movie_id = 1
    for filename in train_filename:
        file = open(filename, 'r')
        retrieval_re = re.compile(r'(\d+),(\d+),.?')
        time_record = -time.time()
        print('Reading file %s ...' % filename)
        for line in file.readlines():
            str = line.strip()
            if str[len(str) - 1] == ':':
                movie_id = int(str[0: len(str) - 1])
                movie_user[movie_id] = []
                movie_user_score[movie_id] = []
            else:
                gp = retrieval_re.match(str).groups()
                cand_user_id = int(gp[0])
                try:
                    user_id = user_map[cand_user_id]
                except:
                    num_users += 1
                    user_map[cand_user_id] = num_users
                    user_id = num_users
                    user_movie[user_id] = []
                    user_movie_score[user_id] = []
                corpus.append((user_id, movie_id, int(gp[1])))
                movie_user[movie_id].append(user_id)
                movie_user_score[movie_id].append(int(gp[1]))
                user_movie[user_id].append(movie_id)
                user_movie_score[user_id].append(int(gp[1]))

        time_record += time.time()
        print('Finished. (Totally %.1f min)' % (time_record / 60))

    print('Reading Finished. Totally %d users, %d films\n'.format(num_users,
                                                                  movie_id))

    tot_movie = movie_id

    test_corpus = []
    file = open(test_filename, 'r')
    retrieval_re = re.compile(r'(\d+),.?')
    movie_id = 1
    for line in file.readlines():
        str = line.strip()
        if str[len(str) - 1] == ':':
            movie_id = int(str[0: len(str) - 1])
        else:
            gp = retrieval_re.match(str).groups()
            cand_user_id = int(gp[0])
            user_id = user_map[cand_user_id]
            test_corpus.append((user_id, movie_id))

    corpus_data = np.array(corpus)
    np.random.shuffle(corpus_data)
    N = np.shape(corpus_data)[1]
    print(N)
    Ndv = N // 10 * 8
    train = corpus_data[:Ndv, :]
    valid = corpus_data[Ndv:, :]
    return tot_movie, num_users, user_map, train, valid, \
        np.array(test_corpus), user_movie, user_movie_score, movie_user, \
        movie_user_score


def read_data_from_csv():
    train_file = 'train'
    test_file = 'test'

    file = open(train_file, 'r')
    retrieval_re = re.compile(r'(\d+),(\d+),(\d+)')

    corpus = []
    num_users = 0
    num_movies = 0
    movie_user = {}
    user_movie = {}
    movie_user_score = {}
    user_movie_score = {}
    for i in range(10000):
        movie_user[i] = []
        user_movie[i] = []
        movie_user_score[i] = []
        user_movie_score[i] = []

    for line in file.readlines():
        str = line.strip()
        gp = retrieval_re.match(str).groups()
        user_id = int(gp[0])
        movie_id = int(gp[1])
        score = int(gp[2])
        if user_id > 100 or movie_id > 50:
            continue
        corpus.append((user_id, movie_id, score))
        movie_user[movie_id].append(user_id)
        movie_user_score[movie_id].append(score)
        user_movie[user_id].append(movie_id)
        user_movie_score[user_id].append(score)
        num_users = max(num_users, user_id)
        num_movies = max(num_movies, movie_id)

    test_corpus = []
    file = open(test_file, 'r')
    for line in file.readlines():
        str = line.strip()
        gp = retrieval_re.match(str).groups()
        user_id = int(gp[0])
        movie_id = int(gp[1])
        score = int(gp[2])
        if user_id > 100 or movie_id > 50:
            continue
        test_corpus.append((user_id, movie_id, score))

    return num_movies, num_users, None, np.array(corpus), None, \
        np.array(test_corpus), user_movie, user_movie_score, movie_user, \
        movie_user_score


def symmetrize(x):
    return 0.5 * (x + tf.transpose(x, perm=[0, 2, 1]))


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
    M, N, user_map, train_data, valid_data, test_data, user_movie, \
        user_movie_score, movie_user, movie_user_score = read_data_from_csv()
    # set configurations and hyper parameters
    N_train = np.shape(train_data)[0]
    N_test = np.shape(test_data)[0]
    D = 10
    epoches = 30
    batch_size = 100
    test_batch_size = 100000
    lambda_U = 0.002
    lambda_V = 0.002
    learning_rate = 0.005
    K = 8
    num_steps = 100000
    save_freq = 1
    iters = (N_train + batch_size - 1) // batch_size
    test_iters = (N_test + test_batch_size - 1) // test_batch_size
    result_path = 'tmp/pmf_map/'

    select_u = tf.placeholder(tf.int32, shape=[None, ], name='s_u')
    select_v = tf.placeholder(tf.int32, shape=[None, ], name='s_v')
    subselect_u = tf.placeholder(tf.int32, shape=[None, ], name='ss_u')
    subselect_v = tf.placeholder(tf.int32, shape=[None, ], name='ss_v')
    alpha_u = 1.0
    alpha_v = 1.0
    alpha_pred = 0.2
    n_leapfrogs = 10

    n = tf.placeholder(tf.int32, shape=[], name='n')
    m = tf.placeholder(tf.int32, shape=[], name='m')

    def log_joint(observed):
        model, _ = pmf(observed, n, m, D, K, subselect_u,
                       subselect_v, alpha_u, alpha_v, alpha_pred)
        log_pu, log_pv = model.local_log_prob(['u', 'v'])    # [K, N], [K, M]
        log_pr = model.local_log_prob('r')                   # [K, batch]
        log_pu = tf.reduce_sum(log_pu, axis=1)
        log_pv = tf.reduce_sum(log_pv, axis=1)
        log_pr = tf.reduce_sum(log_pr, axis=1)
        return log_pu + log_pv + log_pr

    U = tf.get_variable('u', shape=[K, N, D],
                             initializer=tf.random_normal_initializer(0, 0.01),
                             trainable=True)
    V = tf.get_variable('v', shape=[K, M, D],
                             initializer=tf.random_normal_initializer(0, 0.01),
                             trainable=True)

    true_rating = tf.placeholder(tf.float32, shape=[None, ],
                                 name='true_rating')
    true_rating_2d = tf.tile(tf.expand_dims(true_rating, 0), [K, 1])
    fmodel, pred_rating = pmf({'u': U, 'v': V, 'r': true_rating_2d},
                              N, M, D, K,
                              select_u, select_v, alpha_u, alpha_v, alpha_pred)

    pred_rating = tf.reduce_mean(pred_rating, axis=0)
    error = pred_rating - true_rating
    rmse = tf.sqrt(tf.reduce_mean(error * error))
    hmc = zs.HMC(step_size=1e-3, n_leapfrogs=10, adapt_step_size=None,
                   target_acceptance_rate=0.9)
    sample_op, sample_info = \
        hmc.sample(log_joint, {'r': true_rating_2d}, {'u': U, 'v': V})


    mmodel, _ = pmf({'u': U, 'v': V, 'r': true_rating_2d}, n, m, D, K,
                       subselect_u, subselect_v, alpha_u, alpha_v, alpha_pred)
    mlog_pu, mlog_pv, mlog_pr = mmodel.local_log_prob(['u', 'v', 'r'])
    cost = -tf.reduce_sum(mlog_pu) - tf.reduce_sum(mlog_pv) - \
           tf.reduce_sum(mlog_pr)
    optimizer = tf.train.AdamOptimizer(1e-3 * 5, beta1=0.5)
    grads = optimizer.compute_gradients(cost)
    infer = optimizer.apply_gradients(grads)


    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    print('Initialization Finished !')
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
                su = test_data[t * test_batch_size:ed_pos, 0] - 1
                sv = test_data[t * test_batch_size:ed_pos, 1] - 1
                tr = test_data[t * test_batch_size:ed_pos, 2]
                re = sess.run(rmse, feed_dict={select_u: su, select_v: sv,
                                               true_rating: tr,
                                               n: n_batch,
                                               m: n_batch})
                test_rmse.append(re)
            time_test += time.time()
            print('>>> VALIDATION ({:.1f}s)'.format(time_test))
            print('>> Validation rmse = {}'.format(
                (np.mean(test_rmse))))

            epoch_time = -time.time()
            o_u = sess.run(U)
            _, step_size = sess.run([sample_op, sample_info.updated_step_size], feed_dict={n: N, m: M,
                                               true_rating: train_data[:, 2],
                                               subselect_u: train_data[:, 0] - 1,
                                               subselect_v: train_data[:, 1] - 1})
            print('step_size:', step_size)
            n_u = sess.run(U)
            d_u = n_u - o_u
            print(np.sum(d_u * d_u))
            '''
            _ = sess.run(infer, feed_dict={n: N, m: M,
                                               true_rating: train_data[:, 2],
                                               subselect_u: train_data[:, 0] - 1,
                                               subselect_v: train_data[:, 1] - 1})'''
            epoch_time += time.time()
            # exit(0)
            print('Step %d(%.1fs): Finished!' % (step, epoch_time))

            '''
            time_epoch = -time.time()
            res = []
            np.random.shuffle(train_data)
            for t in range(iters):
                ed_pos = min((t + 1) * batch_size, N_train + 1)
                su = train_data[t * batch_size:ed_pos, 0] - 1
                sv = train_data[t * batch_size:ed_pos, 1] - 1
                tr = train_data[t * batch_size:ed_pos, 2]
                _, re = sess.run([infer, rmse],
                                 feed_dict={select_u: su,
                                            select_v: sv,
                                            true_rating: tr})
                res.append(re)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): train rmse = {}'.format(
                step, time_epoch, np.sqrt(np.mean(res))))'''
