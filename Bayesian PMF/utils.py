#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_density_from_data(data, u_en, v_en, moive_cand, user_cand):
    collection = [[], [], [], [], []]
    for data_id in range(np.shape(data)[0]):
        rating = data[data_id, 2]
        u_id = data[data_id, 0]
        v_id = data[data_id, 1]
        if (u_id in user_cand) and (v_id in moive_cand):
            collection[rating - 1].append(np.sum(u_en[u_id, :]
                                                 * v_en[v_id, :]))
    # Padding
    ml = 0
    for i in range(5):
        ml = max(len(collection[i]), ml)
    for i in range(5):
        for j in range(ml - len(collection[i])):
            collection[i].append(None)
    df = pd.DataFrame()
    for i in range(5):
        print(i)
        df['score_{0}'.format(i + 1)] = np.array(collection[i])
    for s in df.columns:
        df[s].plot(kind='density')
    plt.show()
