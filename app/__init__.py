from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db = SQLAlchemy(app)

    with app.app_context():
        from .home import home
        from .project import project

        app.register_blueprint(home.home_bp)
        app.register_blueprint(project.project_bp)

        return app
