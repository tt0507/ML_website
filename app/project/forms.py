from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField
from wtforms.validators import DataRequired
import pandas as pd
import os


class RecommendationForm(FlaskForm):
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/recommendation_cosine/movie_title.csv"))
    movie_choices = list(enumerate(data['original_title']))

    movie_name = SelectField(choices=movie_choices, validators=[DataRequired()])
    submit = SubmitField("Submit")
