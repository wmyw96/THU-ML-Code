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
            else:
                gp = retrieval_re.match(str).groups()
                cand_user_id = int(gp[0])
                try:
                    user_id = user_map[cand_user_id]
                except:
                    num_users += 1
                    user_map[cand_user_id] = num_users
                    user_id = num_users
                corpus.append((user_id, movie_id, int(gp[1])))
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
    return tot_movie, num_users, user_map, train, valid, np.array(test_corpus)


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

    print('Totally %d movies, %d users.' % (num_movies, num_users))
    return num_movies, num_users, None, np.array(corpus), None, \
        np.array(test_corpus), user_movie, user_movie_score, movie_user, \
        movie_user_score


def output_testfile(test_data, pred):
    output_filename = 'output.txt'
    file = open(output_filename, 'w')
    for i in range(np.shape(pred)[0]):
        if i == 0:
            file.write('1:\n')
        else:
            if test_data[i, 1] != test_data[i - 1, 1]:
                file.write('%d:\n' % (test_data[i, 1]))
        file.write('%.3f\n' % pred[i])
    file.close()


if __name__ == '__main__':
    M, N, user_map, train_data, test_data, valid_data, nla, nlb, nlc, nld = \
        read_data_from_csv()

    # set configurations and hyper parameters
    N_train = np.shape(train_data)[0]
    N_valid = np.shape(valid_data)[0]
    D = 30
    epoches = 30
    batch_size = 100000
    valid_batch_size = 100000
    lambda_U = 0.002
    lambda_V = 0.002
    learning_rate = 0.005
    save_freq = 1
    iters = (N_train + batch_size - 1) // batch_size
    valid_iters = (N_valid + valid_batch_size - 1) // valid_batch_size
    result_path = 'tmp/pmf_map/'

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
    pred_rating = tf.reduce_sum(optimized_U * optimized_V, axis=1)

    error = pred_rating - true_rating
    rmse = tf.reduce_mean(error * error)
    cost = 0.5 * tf.reduce_sum(error * error) * \
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

        # Get test answer
        #pred = sess.run(pred_rating, feed_dict={pair_U: test_data[:, 0] - 1,
        #                                        pair_V: test_data[:, 1] - 1})
        #output_testfile(test_data, pred)

        for epoch in range(begin_epoch, epoches + 1):
            time_epoch = -time.time()
            res = []
            np.random.shuffle(train_data)
            for t in range(iters):
                ed_pos = min((t + 1) * batch_size, N_train + 1)
                su = train_data[t * batch_size:ed_pos, 0] - 1
                sv = train_data[t * batch_size:ed_pos, 1] - 1
                tr = train_data[t * batch_size:ed_pos, 2]
                _, re = sess.run([infer, rmse],
                                 feed_dict={pair_U: su,
                                            pair_V: sv,
                                            true_rating: tr,
                                            learning_rate_ph: learning_rate})
                res.append(re)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): train rmse = {}'.format(
                epoch, time_epoch, np.sqrt(np.mean(res))))

            test_rmse = []
            time_test = -time.time()
            for t in range(valid_iters):
                ed_pos = min((t + 1) * valid_batch_size, N_valid + 1)
                su = valid_data[t * valid_batch_size:ed_pos, 0] - 1
                sv = valid_data[t * valid_batch_size:ed_pos, 1] - 1
                tr = valid_data[t * valid_batch_size:ed_pos, 2]
                re = sess.run(rmse, feed_dict={pair_U: su, pair_V: sv,
                                               true_rating: tr})
                test_rmse.append(re)
            time_test += time.time()
            print('>>> VALIDATION ({:.1f}s)'.format(time_test))
            print('>> Validation rmse = {}'.format(np.sqrt(np.mean(test_rmse))))

            if epoch % save_freq == 0:
                #print('Saving model...')
                save_path = os.path.join(result_path,
                                         "pmf_map.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                saver.save(sess, save_path)
                #print('Done')