import googleapiclient.discovery
import json
import os
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler

# load env variable
load_dotenv()

# Used to connect to database
url = os.environ.get("MYSQL_URL")
database = create_engine(url)
mysql = database.connect()
mysql.execute("use ml_website_database")


def predict_next_prices():
    """
    Predict time series using Google Cloud Platform
    :return: predicted values
    """
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = "tidal-eon-283914"

    # dji model
    dji_model_id = "dji_model"
    dji_version = "v0001"
    dji_model_path = "projects/{}/models/{}/versions/{}".format(project_id, dji_model_id, dji_version)

    # sp500 model
    sp500_model_id = "sp500_model"
    sp500_version = "v0001"
    sp500_model_path = "projects/{}/models/{}/versions/{}".format(project_id, sp500_model_id, sp500_version)

    # nasdaq model
    nasdaq_model_id = "nasdaq_model"
    nasdaq_version = "v0001"
    nasdaq_model_path = "projects/{}/models/{}/versions/{}".format(project_id, nasdaq_model_id, nasdaq_version)

    # get the projects
    ml_resource = googleapiclient.discovery.build("ml", "v1").projects()

    # get data for prediction
    dji_data, sp500_data, nasdaq_data = get_data()

    # predict
    dji_predict = predict(dji_data, ml_resource, dji_model_path)
    sp500_predict = predict(sp500_data, ml_resource, sp500_model_path)
    nasdaq_predict = predict(nasdaq_data, ml_resource, nasdaq_model_path)

    # unstandardize the data
    dji_unstandarize = unstandardized(dji_predict, 'dji')
    sp500_unstandarize = unstandardized(sp500_predict, 'sp500')
    nasdaq_unstandarize = unstandardized(nasdaq_predict, 'nasdaq')

    dji = [price['dense_1'] for price in dji_unstandarize]
    sp500 = [price['dense_2'] for price in sp500_unstandarize]
    nasdaq = [price['dense_3'] for price in nasdaq_unstandarize]

    return dji, sp500, nasdaq


def predict(X, resource, model_path):
    """
    Predict next 10 values of time series model
    :param X: data used for prediction
    :param resource: resource object used to call the prediction service
    :param model_path: path for the model
    :return: predicted value as numpy array
    """
    input_data_json = {
        "signature_name": "serving_default",
        "instances": X.tolist()
    }

    request = resource.predict(name=model_path, body=input_data_json)
    response = request.execute()

    if "error" in response:
        raise RuntimeError(response["error"])

    return np.array([pred for pred in response["predictions"]])


def get_data():
    """
    Get the data used for prediction
    :return:
    """
    # Get data from database
    dji_data = mysql.execute("SELECT * FROM ml_website_database.dow_jones_industrial").fetchall()
    sp500_data = mysql.execute("SELECT * FROM ml_website_database.sp500").fetchall()
    nasdaq_data = mysql.execute("SELECT * FROM ml_website_database.nasdaq").fetchall()

    # remove the dates from the fetched data
    dji_numpy = np.array(dji_data)[:, 1:]
    sp500_numpy = np.array(sp500_data)[:, 1:]
    nasdaq_numpy = np.array(nasdaq_data)[:, 1:]

    # Standardize the data
    standard_scaler = StandardScaler()
    dji_standarized = standard_scaler.fit_transform(dji_numpy)
    sp500_standarized = standard_scaler.fit_transform(sp500_numpy)
    nasdaq_standarized = standard_scaler.fit_transform(nasdaq_numpy)

    # transform data to put into RNN
    dji_formatted = format_predict_data(dji_standarized, len(dji_standarized) - 10)
    sp500_formatted = format_predict_data(sp500_standarized, len(sp500_standarized) - 10)
    nasdaq_formatted = format_predict_data(nasdaq_standarized, len(nasdaq_standarized) - 10)
    assert dji_formatted.shape == (10, 20, 6)  # test shape

    return dji_formatted, sp500_formatted, nasdaq_formatted


def format_predict_data(dataset, start_index):
    """
    Create data to be use to predict using RNN
    :param start_index: Starting index to create data
    :param dataset: the dataset used to prediction
    :return: numpy array
    """
    valid_data = []

    # set index
    start_index = start_index
    end_index = len(dataset)

    for i in range(start_index, end_index):
        indices = range(i - 20, i)
        valid_data.append(dataset[indices])

    return np.array(valid_data)


def unstandardized(predicted, table_name):
    """
    Unstandarize the data
    :param table_name:
    :param predicted: Data to be unstandarized
    :return: Data returned as numpy array
    """
    if table_name == 'dji':
        adj_close = mysql.execute("SELECT adj_close FROM ml_website_database.dow_jones_industrial").fetchall()
        adj_close = np.array(adj_close)
    elif table_name == 'sp500':
        adj_close = mysql.execute("SELECT adj_close FROM ml_website_database.sp500").fetchall()
        adj_close = np.array(adj_close)
    else:
        adj_close = mysql.execute("SELECT adj_close FROM ml_website_database.nasdaq").fetchall()
        adj_close = np.array(adj_close)

    mean = adj_close.mean()
    std = adj_close.std()

    for dict_pred in predicted:
        for key, value in dict_pred.items():
            dict_pred[key] = value[0] * std + mean

    return predicted


def get_visualization_data():
    """
    Get data to be used for d3 visualization
    :return: json
    """
    # Get last 10 data from database
    dji_visualization = mysql.execute(
        "SELECT * FROM (SELECT * FROM ml_website_database.dow_jones_industrial ORDER BY date DESC LIMIT 10)"
        "sub ORDER BY date").fetchall()

    sp500_visualization = mysql.execute(
        "SELECT * FROM (SELECT * FROM ml_website_database.sp500 ORDER BY date DESC LIMIT 10)"
        "sub ORDER BY date").fetchall()

    nasdaq_visualization = mysql.execute(
        "SELECT * FROM (SELECT * FROM ml_website_database.nasdaq ORDER BY date DESC LIMIT 10)"
        "sub ORDER BY date").fetchall()

    header = ("date", "open", "high", "low", "close", "adj_close", "volume")
    dji_json = jsonify_data(header, dji_visualization)
    sp500_json = jsonify_data(header, sp500_visualization)
    nasdaq_json = jsonify_data(header, nasdaq_visualization)

    return dji_json, sp500_json, nasdaq_json


def jsonify_data(header, data):
    """
    Turn mysql data into json format
    :param header: header of database
    :param data: data of database
    :return: return json version of data
    """
    json_data = []
    # for row in data:
    #     json_data.append(dict(zip(header, row)))
    for row in data:
        json_data.append({
            header[0]: str(row[0]), header[1]: row[1], header[2]: row[2], header[3]: row[3], header[4]: row[4],
            header[5]: row[5], header[6]: row[6]
        })
    return json.dumps(json_data)


if __name__ == "__main__":
    predict_next_prices()
    # get_visualization_data()
