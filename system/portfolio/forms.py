# FRE_Platform/portfolio/forms.py

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, Length, EqualTo, Email


class RegisterForm(FlaskForm):
    #username = StringField('User Name', validators=[DataRequired(),  Length(min=6, max=40)])
    email = StringField('Email Address', validators=[DataRequired(), Email(), Length(min=6, max=40)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6, max=40)])
    confirm = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])
    last_name = StringField('Last Name', validators=[DataRequired(), Length(min=1, max=40)])
    first_name = StringField('First Name', validators=[DataRequired(), Length(min=1, max=40)])


class LoginForm(FlaskForm):
    #username = PasswordField('User Name', validators=[DataRequired()])
    email = StringField('Email Address', validators=[DataRequired(), Email(), Length(min=6, max=40)])
    password = PasswordField('Password', validators=[DataRequired()])


class EmailForm(FlaskForm):
    email = StringField('Email Address', validators=[DataRequired(), Email(), Length(min=6, max=40)])


class PasswordForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])