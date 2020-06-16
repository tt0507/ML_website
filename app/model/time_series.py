import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
from pandas.io.json import json_normalize


def run_time_series(data1, data2, data3):
    """
    run RNN time series model
    :param data1: dji data
    :param data2: sp500 data
    :param data3: nasdaq data
    :return:
    """
    # import json data and change to dataframe
    dji = json_normalize(json.loads(data1))
    sp500 = json_normalize(json.loads(data2))
    nasdaq = json_normalize(json.loads(data3))

    # set index to date
    dji = dji.set_index('date')
    sp500 = sp500.set_index('date')
    nasdaq = nasdaq.set_index('date')

    assert len(dji) == len(sp500)
    assert len(dji) == len(nasdaq)
    assert len(sp500) == len(nasdaq)

    # Standarize the values
    scale = StandardScaler()
    dji_standarized = scale.fit_transform(dji.values)
    sp500_standarized = scale.fit_transform(sp500.values)
    nasdaq_standarized = scale.fit_transform(nasdaq.values)


def train_data(dataset, target_index, start_index):
    """

    :param dataset: dataset to partition data
    :param target_index: column which to predict
    :param start_index: index to split the data to fit RNN input shape
    :return: train data and label data in numpy array
    """
    data = []
    labels = []
    target = dataset[:, target_index]

    # model will be given 20 prices
    start_index = start_index

    for i in range(start_index, len(dataset)):
        indices = range(i - start_index, i)
        data.append(dataset[indices])
        labels.append(target[i])

    return np.array(data), np.array(labels)


if __name__ == "__main__":
    run_time_series()
