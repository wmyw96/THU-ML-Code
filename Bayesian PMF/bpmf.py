#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
import re
import tensorflow as tf
import numpy as np

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

    print('Reading Finished. Totally %d users, %d films\n'.format(num_users, movie_id))

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
        test_corpus.append((user_id, movie_id, score))

    return num_movies, num_users, None, np.array(corpus), None, \
        np.array(test_corpus), user_movie, user_movie_score, movie_user, \
        movie_user_score


def model_graph(U_t, V_t, N, K):
    beta_0 = 1
    mu_0 = 0
    v_0 = D
    alpha = 2.0
    ds = tf.contrib.distributions

    W_0 = tf.eye(D, batch_shape=[K])

    U_bar = tf.reduce_mean(U_t, axis=1, keep_dims=True)    # (K, 1, D)
    print(U_bar.get_shape())
    S_U = tf.matmul(U_t, U_t, transpose_a=True) / N
    print(S_U.get_shape())
    W_U = tf.matrix_inverse(W_0) + N * S_U + \
          (beta_0 * N + 0.0) / (beta_0 + N) * \
          tf.matmul((mu_0 - U_bar), (mu_0 - U_bar), transpose_a=True)
    W_U = tf.matrix_inverse(W_U)
    print(W_U.get_shape())
    v_U = v_0 + N
    beta_U = beta_0 + N

    v_U = tf.convert_to_tensor(v_U, dtype=tf.float32)
    v_U = tf.tile(tf.expand_dims(v_U, 0), [K])
    Wishart_U = ds.WishartFull(df=v_U, scale=W_U)
    Gamma_U = Wishart_U.sample()      # (K, D, D)

    mu_prior_U = beta_0 * mu_0 + N * tf.squeeze(U_bar, axis=1)
    cov_prior_U = tf.matrix_inverse(beta_U * Gamma_U)
    Gaussian_prior_U = \
        ds.MultivariateNormalFullCovariance(loc = mu_prior_U,
                                            covariance_matrix=cov_prior_U)
    mu_U = Gaussian_prior_U.sample()  # (K, D)
    bias_U = tf.matmul(Gamma_U, tf.expand_dims(mu_U, 2))

    UV_ph = []
    US_ph = []
    U_mus = []
    U_covs = []
    for i in range(N):
        print('build graph: %d' % i)
        select_UiV = tf.placeholder(tf.int32, shape=[None, ],
                                    name='UiV%d' % i)
        select_UiS = tf.placeholder(tf.float32, shape=[None, ],
                                    name='UiS%d' % i)
        UV_ph.append(select_UiV)
        US_ph.append(select_UiS)
        selected_V = tf.gather_nd(V_t,
                                  tf.tile(tf.expand_dims(select_UiV, 0),
                                          [K, 1]))    # [K, num, D]
        gamma_i = Gamma_U + alpha * tf.matmul(selected_V, selected_V,
                                              transpose_a = True)
        cov_i = tf.matrix_inverse(gamma_i) # [K, D, D]
        mu_i = alpha * selected_V * tf.tile(tf.expand_dims(select_UiS, 0),
                                            [K, 1]) * selected_V  # [K, num, D]
        mu_i = tf.expand_dims(tf.reduce_sum(mu_i, axis=1), 2)   # [K, D, 1]
        mu_i = mu_i + bias_U
        mu_i = tf.matmul(cov_i, mu_i)
        mu_i = tf.squeeze(mu_i, 2)
        mu_i = tf.expand_dims(mu_i, 1)
        cov_i = tf.expand_dims(cov_i, 1)
        U_mus.append(mu_i)
        U_covs.append(cov_i)
    U_mu = tf.concat(U_mus, 1)
    U_cov = tf.concat(U_covs, 1)
    Normal_U = \
        ds.MultivariateNormalFullCovariance(loc=U_mu,
                                            covariance_matrix=U_cov)
    U_tp1 = Normal_U.sample()

    return U_tp1, UV_ph, US_ph

if __name__ == '__main__':
    M, N, user_map, train_data, valid_data, test_data, user_movie, \
        user_movie_score, movie_user, movie_user_score = read_data_from_csv()
    # set configurations and hyper parameters
    N_train = np.shape(train_data)[1]
    N_test = np.shape(test_data)[1]
    D = 30
    epoches = 30
    batch_size = 100000
    test_batch_size = 100000
    lambda_U = 0.002
    lambda_V = 0.002
    learning_rate = 0.005
    K = 50
    num_steps = 200
    save_freq = 1
    iters = N_train // batch_size
    test_iters = N_test // test_batch_size
    result_path = 'tmp/pmf_map/'

    U = tf.Variable(random_weights(N, D))     # (N, D)
    V = tf.Variable(random_weights(M, D))     # (M, D)

    U_t = tf.placeholder(tf.float32, shape=[K, N, D], name='U_t')
    V_t = tf.placeholder(tf.float32, shape=[K, M, D], name='V_t')

    U_t0 = tf.tile(tf.expand_dims(U, 0), [K, 1, 1])
    V_t0 = tf.tile(tf.expand_dims(V, 0), [K, 1, 1])

    U_tp1, UV_ph, US_ph = model_graph(U_t, V_t, N, K)
    V_tp1, VU_ph, VS_ph = model_graph(V_t, U_tp1, M, K)

    pair_U = tf.placeholder(tf.int32, shape=[None, ], name='s_U')
    pair_V = tf.placeholder(tf.int32, shape=[None, ], name='s_V')
    true_rating = tf.placeholder(tf.float32, shape=[None, ],
                                 name='true_rating')
    num_films = tf.cast(tf.shape(pair_U)[0], tf.float32)

    optimized_U = tf.gather(U, pair_U)  # (batch_size, D)
    optimized_V = tf.gather(V, pair_V)  # (batch_size, D)
    pred_rating = tf.reduce_sum(optimized_U * optimized_V, axis=1)

    error = pred_rating - true_rating
    rmse = tf.reduce_mean(error * error)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    infer_feed_dict = {}
    for i in range(N):
        infer_feed_dict[UV_ph[i]] = np.array(user_movie[i])
        infer_feed_dict[US_ph[i]] = np.array(user_movie_score[i])
    for i in range(M):
        infer_feed_dict[VU_ph[i]] = np.array(movie_user[i])
        infer_feed_dict[VS_ph[i]] = np.array(movie_user_score[i])

    print('Initialization Finished !')
    saver = tf.train.Saver(max_to_keep=10)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt_file = tf.train.latest_checkpoint(result_path)

        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        U_cur, V_cur = sess.run([U_t0, V_t0])

        for step in range(num_steps):
            epoch_time = -time.time()
            infer_feed_dict[U_t] = U_cur
            infer_feed_dict[V_t] = V_cur
            U_nxt, V_nxt = sess.run([U_tp1, V_tp1],
                                    feed_dict=infer_feed_dict)
            U_cur = U_nxt
            V_cur = V_nxt
            epoch_time += time.time()
            print('Step %d(%.1fs): Finished!' % (step, epoch_time))


