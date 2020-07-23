from flask import Blueprint, render_template, request
from app.data.stock_data import process
from app.model.time_series import predict_next_prices, get_visualization_data
from app.project.forms import RecommendationForm
from app.model.recommendation_cosine import get_recommendation

project_bp = Blueprint(
    'project_bp', __name__,
    template_folder='templates',
    static_folder='static'
)


@project_bp.route('/project/time_forecast')
def time_forecast():

    # get predicted data
    dji_predict_json, sp500_predict_json, nasdaq_predict_json = predict_next_prices()

    # get data for visualization
    dji_json, sp500_json, nasdaq_json = get_visualization_data()

    return render_template('time_forecast.html', dji_json=dji_json, sp500_json=sp500_json, nasdaq_json=nasdaq_json,
                           dji_predict_json=dji_predict_json, sp500_predict_json=sp500_predict_json,
                           nasdaq_predict_json=nasdaq_predict_json)


@project_bp.route('/project/rs_cosine', methods=['GET', 'POST'])
def rs_cosine():
    movie_list = []
    form = RecommendationForm()
    if request.method == "POST":
        movie_id = int(request.form["movie_id"])
        recommendation = get_recommendation(movie_id).values
        movie_list = list(enumerate(recommendation))
    return render_template('rs_cosine.html', form=form, movie_list=movie_list)


@project_bp.route('/project/japan_deflation', methods=['GET'])
def japan_deflation():
    return render_template("deflation_analysis_final.html")
