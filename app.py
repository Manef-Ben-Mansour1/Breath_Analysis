from flask import Flask, request, jsonify, render_template, session
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import bcrypt
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# Configure CORS
CORS(app, supports_credentials=True, origins=["*"], allow_headers=["Content-Type"], methods=["GET", "POST", "OPTIONS"])

# Configure SQLite database
basedir = os.path.abspath(os.path.dirname(__file__))
instance_dir = os.path.join(basedir, 'instance')
os.makedirs(instance_dir, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(instance_dir, "users.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(80), nullable=False)
    last_name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    profile = db.relationship('Profile', backref='user', uselist=False)

# Profile model
class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    height = db.Column(db.Float, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    respiratory_illnesses = db.Column(db.String(200), nullable=False)

# Create database within app context
with app.app_context():
    db.create_all()

# Signup endpoint (user + profile data)
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    email = data.get('email')
    password = data.get('password')
    age = data.get('age')
    height = data.get('height')
    weight = data.get('weight')
    respiratory_illnesses = data.get('respiratory_illnesses')

    if not all([first_name, last_name, email, password,age, height, weight, respiratory_illnesses]):
        return jsonify({'message': 'All fields are required'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'message': 'Email already exists'}), 400

    try:
        height = float(height)
        weight = float(weight)
        age = int(age) if age else None
    except (ValueError, TypeError):
        return jsonify({'message': 'Height and weight must be valid numbers'}), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    new_user = User(first_name=first_name, last_name=last_name, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.flush()  # Get user ID

    new_profile = Profile(
        user_id=new_user.id,
        age=age,
        height=height,
        weight=weight,
        respiratory_illnesses=respiratory_illnesses
    )
    db.session.add(new_profile)
    db.session.commit()

    session['user_id'] = new_user.id
    return jsonify({'message': 'User and profile created successfully', 'user_id': new_user.id}), 201

# Login endpoint
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'message': 'Email and password required'}), 400

    user = User.query.filter_by(email=email).first()
    if user and bcrypt.checkpw(password.encode('utf-8'), user.password):
        session['user_id'] = user.id
        return jsonify({'message': 'Login successful', 'user_id': user.id}), 200
    return jsonify({'message': 'Invalid credentials'}), 401

# Logout endpoint
@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'message': 'Logged out successfully'}), 200

# Profile endpoint (update and retrieve)
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return jsonify({'message': 'Unauthorized'}), 401

    user_id = session['user_id']
    if request.method == 'POST':
        data = request.get_json()
        age = data.get('age')
        height = data.get('height')
        weight = data.get('weight')
        respiratory_illnesses = data.get('respiratory_illnesses')

        if not all([height, weight, age, respiratory_illnesses]):
            return jsonify({'message': 'All profile fields are required'}), 400

        try:
            age = int(age) if age else None
            height = float(height)
            weight = float(weight)
        except (ValueError, TypeError):
            return jsonify({'message': 'Height and weight must be valid numbers'}), 400

        profile = Profile.query.filter_by(user_id=user_id).first()
        if not profile:
            profile = Profile(user_id=user_id)
            db.session.add(profile)

        profile.height = height
        profile.weight = weight
        profile.age = age
        profile.respiratory_illnesses = respiratory_illnesses
        db.session.commit()
        return jsonify({'message': 'Profile updated successfully'}), 200

    profile = Profile.query.filter_by(user_id=user_id).first()
    if profile:
        return jsonify({
            'height': profile.height,
            'weight': profile.weight,
            'age': profile.age,
            'respiratory_illnesses': profile.respiratory_illnesses
        }), 200
    return jsonify({'message': 'Profile not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
