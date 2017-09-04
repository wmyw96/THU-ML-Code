#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
import re
import tensorflow as tf
import numpy as np
from dataset import load_movielens1m_mapped


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


def get_symmetry(x):
    return 0.5 * (x + tf.transpose(x, perm=[0, 2, 1]))


def model_graph(U_t, V_t, N, M, K, mapping, score):
    beta_0 = 2.0
    mu_0 = 0
    v_0 = D
    alpha = 2.0
    ds = tf.contrib.distributions

    W_0 = tf.eye(D, batch_shape=[K])

    U_bar = tf.reduce_mean(U_t, axis=1, keep_dims=True)    # (K, 1, D)
    S_U = tf.matmul(U_t, U_t, transpose_a=True) / N
    W_U = tf.matrix_inverse(W_0) + N * S_U + \
          (beta_0 * N + 0.0) / (beta_0 + N) * \
          tf.matmul((mu_0 - U_bar), (mu_0 - U_bar), transpose_a=True)
    W_U = tf.matrix_inverse(W_U)
    W_U = get_symmetry(W_U)
    v_U = v_0 + N
    beta_U = beta_0 + N

    v_U = tf.convert_to_tensor(v_U, dtype=tf.float32)
    v_U = tf.tile(tf.expand_dims(v_U, 0), [K])
    Wishart_U = ds.WishartFull(df=v_U, scale=W_U)
    Gamma_U = Wishart_U.sample()      # (K, D, D)

    mu_prior_U = (beta_0 * mu_0 + N * tf.squeeze(U_bar, axis=1)) / (beta_0 + N)
    cov_prior_U = tf.matrix_inverse(beta_U * Gamma_U)
    cov_prior_U = get_symmetry(cov_prior_U)
    Gaussian_prior_U = \
        ds.MultivariateNormalFullCovariance(loc=mu_prior_U,
                                            covariance_matrix=cov_prior_U)
    mu_U = Gaussian_prior_U.sample()  # (K, D)
    bias_U = tf.matmul(Gamma_U, tf.expand_dims(mu_U, 2))

    U_mus = []
    U_covs = []
    for i in range(N):
        if i % 10 == 0:
            print('build graph: till node %d' % i)
        select_UiS = tf.convert_to_tensor(score[i], dtype=tf.float32)
        select_UiV = tf.convert_to_tensor(mapping[i], dtype=tf.int32)
        select_UiS = tf.tile(tf.expand_dims(select_UiS, 0), [K, 1])
        select_UiS = tf.expand_dims(select_UiS, 2)

        selected_V = tf.transpose(V_t, perm=[1, 0, 2])
        selected_V = tf.gather(selected_V, select_UiV)
        selected_V = tf.transpose(selected_V, perm=[1, 0, 2])

        gamma_i = Gamma_U + alpha * tf.matmul(selected_V, selected_V,
                                              transpose_a=True)

        cov_i = tf.matrix_inverse(gamma_i) # [K, D, D]
        cov_i = get_symmetry(cov_i)
        mu_i = alpha * selected_V * select_UiS  # [K, num, D]
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
    return U_tp1


if __name__ == '__main__':
    #M, N, user_map, train_data, valid_data, test_data, user_movie, \
    #    user_movie_score, movie_user, movie_user_score = read_data_from_csv()
    M, N, train_data, valid_data, test_data, user_movie, \
        user_movie_score, movie_user, movie_user_score \
            = load_movielens1m_mapped('data/ml-1m.zip')
    # set configurations and hyper parameters
    N_train = np.shape(train_data)[0]
    N_test = np.shape(test_data)[0]
    D = 30
    epoches = 30
    batch_size = 100000
    test_batch_size = 100000
    lambda_U = 0.002
    lambda_V = 0.002
    learning_rate = 0.005
    K = 8
    num_steps = 200
    save_freq = 1
    iters = N_train // batch_size
    test_iters = (N_test + 100000 - 1) // test_batch_size
    result_path = 'tmp/pmf_map/'

    U = tf.Variable(random_weights(N, D))     # (N, D)
    V = tf.Variable(random_weights(M, D))     # (M, D)

    U_t = tf.placeholder(tf.float32, shape=[K, N, D], name='U_t')
    V_t = tf.placeholder(tf.float32, shape=[K, M, D], name='V_t')

    U_t0 = tf.tile(tf.expand_dims(U, 0), [K, 1, 1])
    V_t0 = tf.tile(tf.expand_dims(V, 0), [K, 1, 1])

    U_tp1 = model_graph(U_t, V_t, N, M, K, user_movie, user_movie_score)
    V_tp1 = model_graph(V_t, U_tp1, M, N, K, movie_user, movie_user_score)

    pair_U = tf.placeholder(tf.int32, shape=[None, ], name='s_U')
    pair_V = tf.placeholder(tf.int32, shape=[None, ], name='s_V')
    true_rating = tf.placeholder(tf.float32, shape=[None, ],
                                 name='true_rating')
    num_films = tf.cast(tf.shape(pair_U)[0], tf.float32)

    pred_ratings = []
    for k in range(K):
        optimized_U = tf.gather(U_t[k, :, :], pair_U)  # (batch_size, D)
        optimized_V = tf.gather(V_t[k, :, :], pair_V)  # (batch_size, D)
        pred_ratings.append(tf.reduce_sum(optimized_U * optimized_V, axis=1))
    pred_rating = sum(pred_ratings) / K

    error = pred_rating - true_rating
    rmse = tf.reduce_mean(error * error)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

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

        print('Finish reading map samples')

        for step in range(num_steps):
            test_rmse = []
            time_test = -time.time()
            for t in range(test_iters):
                ed_pos = min((t + 1) * test_batch_size, N_test + 1)
                su = test_data[t * test_batch_size:ed_pos, 0]
                sv = test_data[t * test_batch_size:ed_pos, 1]
                tr = test_data[t * test_batch_size:ed_pos, 2]
                re = sess.run(rmse, feed_dict={pair_U: su, pair_V: sv,
                                               true_rating: tr, U_t: U_cur,
                                               V_t: V_cur})
                test_rmse.append(re)
            time_test += time.time()
            print('>>> TEST ({:.1f}s)'.format(time_test))
            print('>> Test rmse = {}'.format(
                np.sqrt(np.mean(test_rmse))))

            epoch_time = -time.time()
            U_cur, V_cur = sess.run([U_tp1, V_tp1],
                                    feed_dict={U_t: U_cur, V_t: V_cur})
            epoch_time += time.time()
            print('Step %d(%.1fs): Finished!' % (step, epoch_time))