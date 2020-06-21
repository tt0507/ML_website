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
    project_list = [
        {
            'title': 'Time Series Forecasting',
            'description': 'Time series forecasting for Dow Jones Industrial Average, S&P 500, and NASDAQ Composite',
            'link': '/project/time_forecast'
        },
        {
            'title': 'Recommendation System',
            'description': 'Recommendation system for movies based on cosine similarity',
            'link': 'project/rs_cosine'
        }
    ]
    return render_template('project.html', title='Projects', header="Project List", project_list=project_list)


@home_bp.route('/about', methods=['GET'])
def about():
    return render_template('about.html')
