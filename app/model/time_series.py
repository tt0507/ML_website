import googleapiclient.discovery
import json
import os
import numpy as np
from sqlalchemy import create_engine


def predict_next_prices():
    """
    Predict time series using Google Cloud Platform
    :return: predicted values
    """
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

    return dji_predict, sp500_predict, nasdaq_predict


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

    # TODO: check for loop for returning prediction
    return np.array([pred for pred in response["prediction"]])


def get_data():
    """

    :return:
    """
    url = os.environ.get("MYSQL_URL")
    database = create_engine(url)
    mysql = database.connect()
    mysql.execute("use ml_website_database")

    # get last 10 value from database
    dji_data = mysql.execute("SELECT * FROM dow_jones_industrial").fetchall()
    sp500_data = mysql.execute("SELECT * FROM sp500").fetchall()
    nasdaq_data = mysql.execute("SELECT * FROM nasdaq").fetchall()

    header = ("date", "open", "high", "low", "close", "adj_close", "volume")
    dji_json = jsonify_data(header, dji_data)
    sp500_json = jsonify_data(header, sp500_data)
    nasdaq_json = jsonify_data(header, nasdaq_data)

    # TODO: change data into numpy array

    return dji_json, sp500_json, nasdaq_json


def jsonify_data(header, data):
    """
    Turn mysql data into json format
    :param header: header of database
    :param data: data of database
    :return: return json version of data
    """
    json_data = []

    for row in data:
        json_data.append({
            header[0]: str(row[0]), header[1]: row[1], header[2]: row[2], header[3]: row[3], header[4]: row[4],
            header[5]: row[5], header[6]: row[6]
        })
    return json.dumps(json_data)


def get_visualization_data():
    """

    :return:
    """
    pass


if __name__ == "__main__":
    predict_next_prices()
