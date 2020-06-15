from flask import Blueprint, render_template
import app.web_scraping as web_scraping
import pandas as pd

project_bp = Blueprint(
    'project_bp', __name__,
    template_folder='templates',
    static_folder='static'
)


@project_bp.route('/project/time_forecast')
def time_forecast():
    # stock_data = web_scraping.get_stock_price()
    stock_data = [[0, 0], [1, 1], [2, 2]]
    return render_template('time_forecast.html', stock_data=stock_data)
