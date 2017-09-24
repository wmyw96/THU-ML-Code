#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re

import numpy as np
from six.moves import urllib
import zipfile


def download_dataset(url, path):
    print('Downloading data from %s' % url)
    urllib.request.urlretrieve(url, path)


def load_movielens1m(path):
    """
    Loads the movielens 1M dataset.

    :return: The movielens 1M dataset.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset(
            'http://files.grouplens.org/datasets/movielens/ml-1m.zip', path)

    zp = zipfile.ZipFile(path, 'r')
    content = zp.read('ml-1m/ratings.dat')
    data_list = content.split('\n')

    num_users = 0
    num_movies = 0
    corpus = []
    for item in data_list:
        term = item.split('::')
        if len(term) < 3:
            continue
        user_id = int(term[0]) - 1
        movie_id = int(term[1]) - 1
        rating = int(term[2])
        #if movie_id > 50:
        #    continue
        corpus.append((user_id, movie_id, rating))
        num_users = max(num_users, user_id + 1)
        num_movies = max(num_movies, movie_id + 1)

    corpus_data = np.array(corpus)
    np.random.shuffle(corpus_data)
    np.random.shuffle(corpus_data)
    N = np.shape(corpus_data)[0]
    Ndv = N // 20 * 17
    Ndv2 = N // 10 * 9
    train = corpus_data[:Ndv, :]
    valid = corpus_data[Ndv:Ndv2, :]
    test = corpus_data[Ndv2:, :]

    return num_movies, num_users, train, valid, test


def load_movielens1m_mapped(path):
    num_movies, num_users, train, valid, test = load_movielens1m(path)

    user_movie = []
    user_movie_score = []
    for i in range(num_users):
        user_movie.append([])
        user_movie_score.append([])
    movie_user = []
    movie_user_score = []
    for i in range(num_users):
        movie_user.append([])
        movie_user_score.append([])

    for i in range(np.shape(train)[0]):
        user_id = train[i, 0]
        movie_id = train[i, 1]
        rating = train[i, 2]
        user_movie[user_id].append(movie_id)
        user_movie_score[user_id].append(rating)
        movie_user[movie_id].append(user_id)
        movie_user_score[movie_id].append(rating)

    return num_movies, num_users, train, valid, test, \
        user_movie, user_movie_score, movie_user, movie_user_score


def load_movielens1m_mapped_ptest(path, valid_map=False):
    num_movies, num_users, train, valid, test = load_movielens1m(path)

    user_movie = []
    user_movie_score = []
    test_user_movie = []
    test_user_movie_score = []
    valid_user_movie = []
    valid_user_movie_score = []
    for i in range(num_users):
        user_movie.append([])
        user_movie_score.append([])
        test_user_movie.append([])
        test_user_movie_score.append([])
        valid_user_movie.append([])
        valid_user_movie_score.append([])
    movie_user = []
    movie_user_score = []
    for i in range(num_users):
        movie_user.append([])
        movie_user_score.append([])

    for i in range(np.shape(train)[0]):
        user_id = train[i, 0]
        movie_id = train[i, 1]
        rating = train[i, 2]
        user_movie[user_id].append(movie_id)
        user_movie_score[user_id].append(rating)
        movie_user[movie_id].append(user_id)
        movie_user_score[movie_id].append(rating)

    for i in range(np.shape(test)[0]):
        user_id = test[i, 0]
        movie_id = test[i, 1]
        rating = test[i, 2]
        test_user_movie[user_id].append(movie_id)
        test_user_movie_score[user_id].append(rating)

    for i in range(np.shape(valid)[0]):
        user_id = valid[i, 0]
        movie_id = valid[i, 1]
        rating = valid[i, 2]
        valid_user_movie[user_id].append(movie_id)
        valid_user_movie_score[user_id].append(rating)

    if valid_map is False:
        return num_movies, num_users, train, valid, test, \
            user_movie, user_movie_score, movie_user, movie_user_score, \
            test_user_movie, test_user_movie_score
    else:
        return num_movies, num_users, train, valid, test, \
            user_movie, user_movie_score, movie_user, movie_user_score, \
            test_user_movie, test_user_movie_score, valid_user_movie, \
            valid_user_movie_score
