from app import app
from flask import render_template, url_for, redirect


@app.route('/')
def index():
    return render_template('index.html', title='Website', header="Machine Learning Projects")


@app.route('/project')
def project():
    finance = [
        {
            'title': 'Time Series Forecasting',
            'description': 'Time series forecasting for Dow Jones Industrial Average, S&P 500, and NASDAQ Composite',
            'link': '/project/time_forecast'
        }
    ]
    return render_template('project.html', title='Projects', header="Project List", finance=finance)


@app.route('/project/time_forecast')
def project_time_forecast():
    return render_template('project/time_forecast.html')
