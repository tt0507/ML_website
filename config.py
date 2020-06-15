import os


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'secret_key'

    # mysql
    MYSQL_HOST = os.environ.get('MYSQL_HOST')
    MYSQL_USER = os.environ.get('MYSQL_USER')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD')
    MYSQL_DB = os.environ.get('MYSQL_DB')
