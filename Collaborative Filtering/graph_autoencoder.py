#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import numpy as np
from dataset import load_movielens1m
import random


def sparse_dropout(x, keep_prob):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([tf.shape(x.indices)[0]])
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out


def get_traing_data(l, r, idxes, keep_prob, train_data,
                    adj_matrixes, degree_vectors, selected_users,
                    selected_movies, true_ratings, selected_node):
    id_map = {}
    s_nodes = []
    for i in range(l, r):
        id = idxes[i]
        id_map[id] = i - l
        s_nodes.append(id)
    n_node = r - l

    adjms_ind = []
    adjms_val = []
    dgs = []
    for r in range(5):
        adjms_ind.append([])
        adjms_val.append([])
        dgs.append(np.zeros([n_node]))

    s_users = []
    s_movies = []
    s_ratings = []
    for r in range(5):
        for i in range(n_node):
            adjms_ind[r - 1].append((i, i))
            adjms_val[r - 1].append(1)
            dgs[r - 1][i] = 1

    for i in range(np.shape(train_data)[0]):
        id_x = train_data[i, 0]
        id_y = train_data[i, 1]
        rt = train_data[i, 2]
        if (id_x not in id_map) or (id_y not in id_map):
            continue
        id_mx = id_map[id_x]
        id_my = id_map[id_y]
        s_users.append(id_mx)
        s_movies.append(id_my)
        s_ratings.append(rt)
        if np.random.uniform(0, 1.0) > keep_prob:
            continue
        dgs[rt - 1][id_mx] += 1 / keep_prob
        dgs[rt - 1][id_my] += 1 / keep_prob
        adjms_ind[rt - 1].append((id_mx, id_my))
        adjms_ind[rt - 1].append((id_my, id_mx))
        adjms_val[rt - 1].append(1 / keep_prob)
        adjms_val[rt - 1].append(1 / keep_prob)

    dicts = {selected_node: np.array(s_nodes),
             selected_movies: np.array(s_movies),
             selected_users: np.array(s_users),
             true_ratings: np.array(s_ratings)}
    for i in range(5):
        dicts[adj_matrixes[i]] = tf.SparseTensorValue(np.array(adjms_ind[i]),
                                                      np.array(adjms_val[i]),
                                                      [n_node, n_node])
        dicts[degree_vectors[i]] = dgs[i]

    return dicts


def get_test_data(test_epoch, train_data,
                  adj_matrixes, degree_vectors, selected_users,
                  selected_movies, true_ratings, selected_node):
    id_map = {}
    s_nodes = []
    n_node = 0
    s_users = []
    s_movies = []
    s_ratings = []
    for i in range(np.shape(test_epoch)[0]):
        idx = test_epoch[i, 0]
        if idx in id_map:
            pass
        else:
            id_map[idx] = n_node
            n_node += 1
            s_nodes.append(idx)
        idx = test_epoch[i, 1]
        if idx in id_map:
            pass
        else:
            id_map[idx] = n_node
            n_node += 1
            s_nodes.append(idx)
        s_users.append(id_map[test_epoch[i, 0]])
        s_movies.append(id_map[test_epoch[i, 1]])
        s_ratings.append(test_epoch[i, 2])

    adjms_ind = []
    adjms_val = []
    dgs = []
    for r in range(5):
        adjms_ind.append([])
        adjms_val.append([])
        dgs.append(np.zeros([n_node]))

    for r in range(5):
        for i in range(n_node):
            adjms_ind[r - 1].append((i, i))
            adjms_val[r - 1].append(1)
            dgs[r - 1][i] = 1

    for i in range(np.shape(train_data)[0]):
        id_x = train_data[i, 0]
        id_y = train_data[i, 1]
        rt = train_data[i, 2]
        if (id_x not in id_map) or (id_y not in id_map):
            continue
        id_mx = id_map[id_x]
        id_my = id_map[id_y]
        dgs[rt - 1][id_mx] += 1
        dgs[rt - 1][id_my] += 1
        adjms_ind[rt - 1].append((id_mx, id_my))
        adjms_ind[rt - 1].append((id_my, id_mx))
        adjms_val[rt - 1].append(1)
        adjms_val[rt - 1].append(1)

    dicts = {selected_node: np.array(s_nodes),
             selected_movies: np.array(s_movies),
             selected_users: np.array(s_users),
             true_ratings: np.array(s_ratings)}
    for i in range(5):
        dicts[adj_matrixes[i]] = tf.SparseTensorValue(np.array(adjms_ind[i]),
                                                      np.array(adjms_val[i]),
                                                      [n_node, n_node])
        dicts[degree_vectors[i]] = dgs[i]

    return dicts


if __name__ == '__main__':
    np.random.seed(1234)
    tf.set_random_seed(1237)
    N_movie, N_user, train_data, valid_data, test_data \
        = load_movielens1m('data/ml-1m.zip')
    N_node = N_movie + N_user

    train_data[:, 0] += N_movie
    valid_data[:, 0] += N_movie
    test_data[:, 0] += N_movie

    # set configurations and hyper parameters
    N_train = np.shape(train_data)[0]
    N_ratings = 5
    N_test = np.shape(test_data)[0]
    N_valid = np.shape(valid_data)[0]
    n_hidden_l1 = 500
    n_hidden_l2 = 75
    batch_size = 10000
    test_batch_size = 100000
    valid_batch_size = 100000
    num_epochs = 1000
    learning_rate = 1e-2
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    iters = (N_node + batch_size - 1) // batch_size
    test_iters = (N_test + test_batch_size - 1) // test_batch_size
    valid_iters = (N_valid + valid_batch_size - 1) // valid_batch_size
    test_freq = 10
    valid_freq = 10
    accum_stack = False
    constant_keep_dropout = 0.8
    result_path = 'tmp/graph_auto/'

    selected_node = tf.placeholder(tf.int32, shape=[None, ], name='s_node')
    keep_prob = tf.placeholder(tf.float32, shape=[], name='kp')
    n = tf.shape(selected_node)[0]

    Ws = []
    for i in range(N_ratings):
        T = tf.get_variable('t_%d' % i, shape=[N_node, n_hidden_l1],
                            initializer=tf.random_normal_initializer(0, 0.1))
        if i == 0:
            Ws.append(T)
        else:
            Ws.append(Ws[i - 1] + T)

    adj_matrixes = []
    degree_vectors = []
    messages = []
    for i in range(N_ratings):
        adjm = tf.sparse_placeholder(tf.float32, name='adjm_%d' % i,
                                     shape=[None, None])
        deg = tf.placeholder(tf.float32, name='degress_%d' % i, shape=[None])
        adj_matrixes.append(adjm)
        degree_vectors.append(deg)

        input_feature = tf.gather(Ws[i], selected_node)  # [n, n_hidden_l1]
        message = 1.0 / tf.expand_dims(tf.maximum(deg, 1.0), axis=1) * \
            tf.sparse_tensor_dense_matmul(adjm, input_feature)
        messages.append(message)

    if accum_stack is True:
        hidden_1 = tf.concat(messages, axis=1)
        n_hidden_l1 *= N_ratings
    else:
        hidden_1 = sum(messages)
    hidden_1 = tf.nn.relu(hidden_1)
    hidden_1 = tf.nn.dropout(hidden_1, keep_prob=keep_prob)

    W_comm = tf.get_variable('w_comm', shape=[n_hidden_l1, n_hidden_l2],
                             initializer=tf.random_normal_initializer(0, 0.1))
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, W_comm))

    selected_users = tf.placeholder(tf.int32, shape=[None, ],
                                    name='selected_users')
    selected_movies = tf.placeholder(tf.int32, shape=[None, ],
                                     name='selected_movies')
    true_ratings = tf.placeholder(tf.int32, shape=[None, ],
                                  name='true_ratings')
    users_embedding = tf.gather(hidden_2, selected_users)    # [batch, n_h2]
    movies_embedding = tf.gather(hidden_2, selected_movies)  # [batch, n_h2]

    users_embedding_p = \
        tf.tile(tf.expand_dims(users_embedding, axis=0), [N_ratings, 1, 1])
    movies_embedding_p = \
        tf.tile(tf.expand_dims(movies_embedding, axis=0), [N_ratings, 1, 1])
    P = tf.get_variable('P', shape=[n_hidden_l2, n_hidden_l2, 2],
                        initializer=tf.random_normal_initializer(0, 0.1))
    alpha_P = tf.get_variable('alpha_P', shape=[1, 2, N_ratings])
    Q = tf.transpose(tf.matmul(P, tf.tile(alpha_P, [n_hidden_l2, 1, 1])),
                     [2, 0, 1])   # [N_rating, n_h2, n_h2]
    score = \
        tf.reduce_sum(tf.matmul(users_embedding_p, Q) * movies_embedding_p, 2)
    constant_rating = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
    constant_rating = tf.tile(tf.expand_dims(constant_rating, 1),
                              [1, tf.shape(selected_users)[0]])
    prop = tf.nn.softmax(score, dim=0)
    pred = tf.reduce_sum(prop * constant_rating, axis=0)

    error = (pred - tf.cast(true_ratings, dtype=tf.float32))

    se = tf.reduce_sum(error * error)
    rmse = tf.sqrt(tf.reduce_mean(error * error))

    label = tf.one_hot(true_ratings - 1, depth=5)
    logits = tf.transpose(score, [1, 0])
    accuracy = tf.reduce_mean(tf.nn.softmax(logits) * label * 5)
    cost = -tf.reduce_mean(tf.log(tf.nn.softmax(logits) + 1e-8) * label) + 0.0

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, beta1=0.5)
    grads = optimizer.compute_gradients(cost)
    infer = optimizer.apply_gradients(grads)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, num_epochs + 1):
            time_epoch = -time.time()
            res = []
            acc = []
            idxes = range(N_node)
            np.random.shuffle(idxes)
            tot = 0
            for t in range(iters):
                ed_pos = min((t + 1) * batch_size, N_node)
                l = t * batch_size
                r = min((t + 1) * batch_size, N_node)
                cur_batch_size = r - l
                feed_dicts = get_traing_data(l, r, idxes,
                                             constant_keep_dropout,
                                             train_data,
                                             adj_matrixes,
                                             degree_vectors,
                                             selected_users,
                                             selected_movies,
                                             true_ratings,
                                             selected_node)
                feed_dicts[learning_rate_ph] = learning_rate
                feed_dicts[keep_prob] = constant_keep_dropout
                _, __, ___, ____ = sess.run([infer, se, accuracy, label],
                                            feed_dict=feed_dicts)
                res.append(__)
                acc.append(___)
                tot = tot + np.shape(feed_dicts[selected_users])[0]
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): train rmse = {}, acc= {}'.format(
                epoch, time_epoch, np.sqrt(np.sum(res) / tot), np.mean(acc)))

            if epoch % valid_freq == 0:
                valid_rmse = []
                time_valid = -time.time()
                for t in range(valid_iters):
                    ed_pos = min((t + 1) * valid_batch_size, N_valid)
                    bg = t * valid_batch_size
                    feed_dicts = get_test_data(valid_data[bg: ed_pos, :],
                                               train_data,
                                               adj_matrixes,
                                               degree_vectors,
                                               selected_users,
                                               selected_movies,
                                               true_ratings,
                                               selected_node)
                    feed_dicts[keep_prob] = 1.0
                    __ = sess.run(se, feed_dict=feed_dicts)
                    valid_rmse.append(__)
                time_valid += time.time()
                print('>>> VALIDATION ({:.1f}s)'.format(time_valid))
                print('>> Validation rmse = {}'.
                      format(np.sqrt(np.sum(valid_rmse) / N_valid)))
            if epoch % test_freq == 0:
                test_rmse = []
                time_test = -time.time()
                for t in range(test_iters):
                    ed_pos = min((t + 1) * test_batch_size, N_test)
                    bg = t * test_batch_size
                    feed_dicts = get_test_data(test_data[bg: ed_pos, :],
                                               train_data,
                                               adj_matrixes,
                                               degree_vectors,
                                               selected_users,
                                               selected_movies,
                                               true_ratings,
                                               selected_node)
                    feed_dicts[keep_prob] = 1.0
                    __ = sess.run(se, feed_dict=feed_dicts)
                    test_rmse.append(__)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test rmse = {}'.
                      format(np.sqrt(np.sum(test_rmse) / N_test)))

