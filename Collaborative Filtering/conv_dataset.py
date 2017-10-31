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

    output1 = open('train', 'w')
    output2 = open('test', 'w')
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

    for i in range(np.shape(train)[0]):
        output1.write('%d\t%d\t%d\n' % (train[i, 0], train[i, 1], train[i, 2]))
    output1.close()
    for i in range(np.shape(test)[0]):
        output2.write('%d\t%d\t%d\n' % (test[i, 0], test[i, 1], test[i, 2]))
    output2.close()    

    return num_movies, num_users, train, valid, test

if __name__ == "__main__":
    load_movielens1m('data/ml-1m.zip')
    
