
#%%
from wtforms import Form, StringField, PasswordField, validators, SubmitField, SelectField

from wtforms.validators import ValidationError, DataRequired, Email, EqualTo, Length


class SignupForm(Form):
    """User Signup Form."""

    name = StringField('Name', [
        validators.DataRequired(message=('Don\'t be shy!'))
    ])
    email = StringField('Email', [
        Length(min=6, message=(u'Little short for an email address?')),
        Email(message=('That\'s not a valid email address.')),
        DataRequired(message=('That\'s not a valid email address.'))
    ])
    password = PasswordField('Password', validators=[
        DataRequired(message="Please enter a password."),
    ])
    confirmPassword = PasswordField('Repeat Password', validators=[
            EqualTo(password, message='Passwords must match.')
            ])
    website = StringField('Website')
    submit = SubmitField('Register')

    def validate_email(self, email):
        """Email validation."""
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Please use a different email address.')

#%%
