from flask import Flask, request, render_template, redirect, session,jsonify
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from src.pipelines.prediction_pipeline import CustomData,PredictionPipeline
# Initialize the Flask application
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))


with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/frontpage')
        else:
            return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')



@app.route('/predict-custom', methods=['POST'])
def predict_custom():
    data = request.json
    file_path = data.get('file_path')
    # Add logic to process the custom file
    result = {"message": f"Prediction for custom file: {file_path}"}
    return jsonify(result)

@app.route('/predict-default', methods=['GET'])
def predict_default():
    # Add logic to process the default file
    result = {"message": "Prediction for default file completed"}
    return jsonify(result)



@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/login')
 
@app.route('/frontpage',methods=['GET','POST'])
def frontpage():
    if 'email' not in session:
        return redirect('/login')
    
    return render_template('1.html')

@app.route('/predict_datapoint', methods=['GET', 'POST'])
def predict_datapoint():
    if 'email' not in session:
        return redirect('/login')
    
    if request.method=='POST':
       
        data=CustomData(
                  age = float(request.form.get('age')),
                  sex = 1 if request.form.get('sex') == 'male' else 0,
                  on_thyroxine = 1 if request.form.get('on_thyroxine') == 'yes' else 0,
                  sick = 1 if request.form.get('sick') == 'yes' else 0,
                  pregnant = 1 if request.form.get('pregnant') == 'yes' else 0,
                  I131_treatment = 1 if request.form.get('I131_treatment') == 'yes' else 0,
                  query_hypothyroid = 1 if request.form.get('query_hypothyroid') == 'yes' else 0,
                  goitre = 1 if request.form.get('goitre') == 'yes' else 0,
                  psych = 1 if request.form.get('psych') == 'yes' else 0,
                  TSH = float(request.form.get('TSH')),
                  T3 = float(request.form.get('T3')),
                  TT4 = float(request.form.get('TT4')),
                  T4U = float(request.form.get('T4U')),
                  FTI = float(request.form.get('FTI')),
                  referral_source_SVHC = 1 if request.form.get('referral_source') == 'SVHC' else 0,
                  referral_source_other = 1 if request.form.get('referral_source') == 'other' else 0
        )
       
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictionPipeline()
        prediction=predict_pipeline.predict(final_new_data)
        if prediction[0] == 1:
            result = 'Negative'
        elif prediction[0] == 0:
            result = 'Compensated Hypothyroid'
        elif prediction[0] == 2:
            result = 'Primary Hypothyroid'
        else:
            result = 'Secondary Hypothyroid'

      

        return render_template('single_prediction.html',final_result=result)
        
    return render_template('home.html')
    

if __name__=="__main__":
    app.run(host='0.0.0.0',port=5602,debug=True)