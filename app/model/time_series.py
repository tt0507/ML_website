import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
from pandas.io.json import json_normalize
from pathlib import Path


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

    # split to train data and label data
    dji_data, dji_label = train_data(dji_standarized, 4, 20)
    sp500_data, sp500_label = train_data(sp500_standarized, 4, 20)
    nasdaq_data, nasdaq_label = train_data(nasdaq_standarized, 4, 20)

    # check shape of training data
    assert dji_data.shape == (len(dji_standarized) - 20, 20, 6)
    assert sp500_data.shape == (len(sp500_standarized) - 20, 20, 6)
    assert nasdaq_data.shape == (len(nasdaq_standarized) - 20, 20, 6)

    # check shape of label data
    assert dji_label.shape == (len(dji_standarized) - 20,)
    assert sp500_label.shape == (len(sp500_standarized) - 20,)
    assert nasdaq_label.shape == (len(nasdaq_standarized) - 20,)

    dji_model_file = Path("/app/model/saved_model/dji_model.h5")
    if dji_model_file.is_file():
        print(True)
    else:
        print(False)


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


def predict_next_prices(train, label, model, predict_num):
    """
    Train RNN model on training data and label data
    :param predict_num:
    :param train: train data
    :param label: label data
    :param model: path to save model
    :return:
    """
    model = keras.models.Sequential([
        keras.layers.LSTM(100, return_sequences=True, input_shape=train.shape[-2:]),
        keras.layers.LSTM(50, return_sequences=True),
        keras.layers.LSTM(25),
        keras.layers.Dense(1)
    ])

    # fit model
    callback = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    checkpoint_cb = keras.callbacks.ModelCheckpoint(model, save_best_only=True)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
    model.fit(train, label, epochs=100, callbacks=[callback, checkpoint_cb])

    # predict data
    predict_data = train[-predict_num:]
    predict = model.predict(predict_data)

    return predict


if __name__ == "__main__":
    run_time_series()
