from flask import Flask, render_template, request, redirect, url_for, session, flash  
from flask_sqlalchemy import SQLAlchemy 
from werkzeug.security import generate_password_hash, check_password_hash 

app = Flask(__name__)
app.secret_key = 'secretkey'  # secret key for session management
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # to suppress a warning from SQLAlchemy
db = SQLAlchemy(app) # initialize the database

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200)) # Increased to 200 to safely store hashed passwords

# Database initialization with app context
with app.app_context(): 
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

# New route to display your combined login/register page
@app.route('/auth')
def auth():
    return render_template('authentication.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirm_password']

    # validations
    if not name or len(name.strip()) < 2:
        flash('Name must be at least 2 characters long.', 'error')
        return redirect(url_for('auth'))
    
    if not email or '@' not in email:
        flash('Please enter a valid email address.', 'error')
        return redirect(url_for('auth'))
    
    # password must be at least 8 characters long and a combination of letters and numbers and special characters
    if len(password) < 8 or not any(char.isdigit() for char in password)\
          or not any(char.isalpha() for char in password) or not any(not char.isalnum()\
                                                                      for char in password):
        flash('Password must be at least 8 characters long and contain letters, numbers, and special characters.', 'error')
        return redirect(url_for('auth'))
    
    if password != confirm_password:
        flash('Passwords do not match.', 'error')
        return redirect(url_for('auth'))
    
    # check if user already exists
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        flash('Email already registered. Please log in.', 'error')
        return redirect(url_for('auth'))
    
    # create new user
    hashed_password = generate_password_hash(password)
    new_user = User(
        name=name.strip(),
        email=email.strip(),
        password=hashed_password
    )
    try:
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth'))
    except Exception as e:
        db.session.rollback()
        flash('An error occurred during registration. Please try again.', 'error')
        return redirect(url_for('auth'))

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    user = User.query.filter_by(email=email).first()

    if user and check_password_hash(user.password, password):
        session['user_id'] = user.id
        session['user_name'] = user.name
        flash('Login successful!', 'success')
        return redirect(url_for('index'))
    else:
        flash('Invalid email or password.', 'error')
        return redirect(url_for('auth'))

@app.route('/upload',methods=['POST'])
def Upload():
    return render_template('upload.html')

        
if __name__ == '__main__':
    app.run(debug=True)