import json
import os

from flask import Blueprint, render_template, request
from sqlalchemy import create_engine

from app.data.stock_data import process
from app.model.time_series import run_time_series
from app.project.forms import RecommendationForm
from app.model.recommendation_cosine import get_recommendation

project_bp = Blueprint(
    'project_bp', __name__,
    template_folder='templates',
    static_folder='static'
)

url = os.environ.get("MYSQL_URL")
database = create_engine(url)
mysql = database.connect()
mysql.execute("use ml_website_database")


@project_bp.route('/project/time_forecast')
def time_forecast():
    # insert into database
    process()

    # get value from database
    dji_data = mysql.execute("SELECT * FROM dow_jones_industrial").fetchall()
    sp500_data = mysql.execute("SELECT * FROM sp500").fetchall()
    nasdaq_data = mysql.execute("SELECT * FROM nasdaq").fetchall()

    header = ("date", "open", "high", "low", "close", "adj_close", "volume")
    dji_json = jsonify_data(header, dji_data)
    sp500_json = jsonify_data(header, sp500_data)
    nasdaq_json = jsonify_data(header, nasdaq_data)

    dji_predict_json, sp500_predict_json, nasdaq_predict_json = run_time_series(dji_json, sp500_json, nasdaq_json)

    return render_template('time_forecast.html', dji_json=dji_json, sp500_json=sp500_json, nasdaq_json=nasdaq_json,
                           dji_predict_json=dji_predict_json, sp500_predict_json=sp500_predict_json,
                           nasdaq_predict_json=nasdaq_predict_json)


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


@project_bp.route('/project/rs_cosine', methods=['GET', 'POST'])
def rs_cosine():
    form = RecommendationForm()
    if request.method == "POST":
        movie_id = request.form["movie_id"]
    return render_template('rs_cosine.html', form=form)
