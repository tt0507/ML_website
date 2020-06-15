from flask import Blueprint, render_template
import app.web_scraping as web_scraping
import pandas as pd
from sqlalchemy import create_engine
import os

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
    # stock_data = web_scraping.get_stock_price()
    data = mysql.execute("SELECT * FROM dow_jones_industrial")
    return render_template('time_forecast.html', stock_data=data)
