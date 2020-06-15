from app import app
from flask import render_template, url_for, redirect


@app.route('/')
def index():
    return render_template('index.html', title='Website', header="Machine Learning Projects")


@app.route('/project')
def project():
    return render_template('project.html', title='Projects', header="Project List")
