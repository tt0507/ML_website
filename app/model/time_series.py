import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from pandas.io.json import json_normalize
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


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

    # run model and get predicted value
    dji_predict_stan = run_model("app/model/saved_model/dji_model.h5", dji_data, dji_label, 10)
    sp500_predict_stan = run_model("app/model/saved_model/sp500_model.h5", sp500_data, sp500_label, 10)
    nasdaq_predict_stan = run_model("app/model/saved_model/nasdaq_model.h5", nasdaq_data, nasdaq_label, 10)

    # get unstandarized version of predicted value
    dji_predict = unstandarize(dji_predict_stan, dji['adj_close'])
    sp500_predict = unstandarize(sp500_predict_stan, sp500['adj_close'])
    nasdaq_predict = unstandarize(nasdaq_predict_stan, nasdaq['adj_close'])

    dji_predict_json = json.dumps(dict(zip(list(range(len(dji_predict))), dji_predict.tolist())))
    sp500_predict_json = json.dumps(dict(zip(list(range(len(sp500_predict))), sp500_predict.tolist())))
    nasdaq_predict_json = json.dumps(dict(zip(list(range(len(nasdaq_predict))), nasdaq_predict.tolist())))

    return dji_predict_json, sp500_predict_json, nasdaq_predict_json


def train_data(dataset, target_index, start_index):
    """
    split data in 3D array of shape (len(dataset), start_index, # of columns)
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


def predict_next_prices(train, label, model_path, predict_num):
    """
    Train RNN model on training data and label data
    :param model_path:
    :param predict_num:
    :param train: train data
    :param label: label data
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
    checkpoint_cb = keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
    model.fit(train, label, epochs=100, callbacks=[callback, checkpoint_cb])
    model.save(model_path)

    # predict data
    predict_data = train[-predict_num:]
    predict = model.predict(predict_data)

    return predict


def run_model(path, data, label, predict_length):
    """
    Run RNN model
    :param path: path to .h5 model file
    :param data: train data
    :param label: target data
    :param predict_length: amount of days to predict
    :return: predicted value for predict_length days
    """
    model_file = Path(path)
    if model_file.is_file():
        predict_dji = data[-predict_length:]
        model = keras.models.load_model(path)
        predicted_value = model.predict(predict_dji)
    else:
        predicted_value = predict_next_prices(data, label, path,
                                              predict_num=predict_length)
    assert predicted_value.shape == (predict_length, 1)

    return predicted_value


def unstandarize(predict, target_data):
    """
    unstandarize the predict column which represents a Z-score value
    :param predict:
    :param target_data:
    :return:
    """
    mean = target_data.mean()
    std = target_data.std()
    unstandarized_arr = predict * std + mean
    return unstandarized_arr


if __name__ == "__main__":
    run_time_series()
