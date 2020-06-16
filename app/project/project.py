from flask import Blueprint, render_template
import pandas as pd
from sqlalchemy import create_engine
import os
from app.data.stock_data import process

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
    process()
    data = mysql.execute("SELECT * FROM demo").fetchall()  # to check functionality
    # data = mysql.execute("SELECT * FROM dow_jones_industrial").fetchall()
    return render_template('time_forecast.html', stock_data=data)
