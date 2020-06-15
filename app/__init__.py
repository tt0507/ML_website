from flask import Flask
from config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    with app.app_context():
        from .home import home
        from .project import project

        app.register_blueprint(home.home_bp)
        app.register_blueprint(project.project_bp)

        return app
