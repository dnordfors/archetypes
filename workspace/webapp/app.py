from flask import Flask, url_for, render_template, redirect, request, flash
from forms import SignupForm
import config
import sys
import json


app = Flask(__name__, static_url_path='', static_folder="static", template_folder="templates")
compress = FlaskStaticCompress(app)
app.config.from_object('config.Config')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup Form."""
    signup_form = SignupForm()
    if request.method == 'POST':
        if signup_form.validate():
            flash('Logged in successfully.')
            return render_template('/dashboard.html', template="dashbord-template")
    return render_template('/signup.html', form=signup_form, template="form-page")