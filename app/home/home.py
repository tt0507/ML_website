from flask import Blueprint, render_template
from flask import current_app as app

# blueprint configuration
home_bp = Blueprint(
    'home_bp', __name__,
    template_folder='templates',
    static_folder='static'
)


@home_bp.route('/', methods=['GET'])
def index():
    return render_template('index.html', title='Website', header="Machine Learning Projects")


@home_bp.route('/project', methods=['GET'])
def project():
    finance = [
        {
            'title': 'Time Series Forecasting',
            'description': 'Time series forecasting for Dow Jones Industrial Average, S&P 500, and NASDAQ Composite',
            'link': '/project/time_forecast'
        }
    ]
    return render_template('project.html', title='Projects', header="Project List", finance=finance)
