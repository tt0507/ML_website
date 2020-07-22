import googleapiclient.discovery
import json
import os
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine

# load env variable
load_dotenv()


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
        "instances": [X.tolist()]
    }

    request = resource.predict(name=model_path, body=input_data_json)
    response = request.execute()

    if "error" in response:
        raise RuntimeError(response["error"])

    # TODO: check for loop for returning prediction
    return np.array([pred for pred in response["predictions"]])


def get_data():
    """
    Get the data used for prediction
    :return:
    """

    url = os.environ.get("MYSQL_URL")
    database = create_engine(url)
    mysql = database.connect()
    mysql.execute("use ml_website_database")

    # get last 10 value from database
    dji_data = mysql.execute(
        "SELECT * FROM ( SELECT * FROM ml_website_database.dow_jones_industrial ORDER BY date DESC LIMIT 10) sub "
        "ORDER BY date ").fetchall()
    sp500_data = mysql.execute(
        "SELECT * FROM ( SELECT * FROM ml_website_database.sp500 ORDER BY date DESC LIMIT 10) sub "
        "ORDER BY date ").fetchall()
    nasdaq_data = mysql.execute(
        "SELECT * FROM ( SELECT * FROM ml_website_database.nasdaq ORDER BY date DESC LIMIT 10) sub "
        "ORDER BY date ").fetchall()

    dji_numpy = np.array(dji_data)[:, 1:]
    sp500_numpy = np.array(sp500_data)[:, 1:]
    nasdaq_numpy = np.array(nasdaq_data)[:, 1:]

    return dji_numpy, sp500_numpy, nasdaq_numpy


def get_visualization_data():
    """

    :return:
    """
    pass


if __name__ == "__main__":
    predict_next_prices()
