from flask import Blueprint, render_template
from flask import current_app as app

project_bp = Blueprint(
    'project_bp', __name__,
    template_folder='templates',
    static_folder='static'
)


@project_bp.route('/project/time_forecast')
def time_forecast():
    return render_template('time_forecast.html')
