from flask import Flask, request, render_template, redirect, session,jsonify,send_file
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import os

from src.components.data_ingestion import Data_Ingestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer 
from src.pipelines.prediction_pipeline import CustomData,PredictionPipeline
import pandas as pd
import pickle
# Initialize the Flask application
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'
preprocessor_path=r"C:\Users\ys136\Desktop\Data Science\End to End ML Projects\thyroid_Prediction_System\artifacts\preprocessor.pkl"
model=r"C:\Users\ys136\Desktop\Data Science\End to End ML Projects\thyroid_Prediction_System\artifacts\model.pkl"
preprocessor=pickle.load(open(preprocessor_path,'rb'))
model=pickle.load(open(model,'rb'))

def preprocess_data(df):
    # Select relevant features (Ensure column names match training data)
    required_columns = [ "age", "sex", "on_thyroxine", "sick", "pregnant", "I131_treatment",
            "query_hypothyroid", "goitre", "psych", "TSH", "T3", "TT4", "T4U", "FTI",
            "referral_source_SVHC", "referral_source_other"]  # Replace with actual feature names
    df = df[required_columns]

    df=preprocessor.transform(df)
    
    return df


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
    try:
        file_path = request.form.get('filepath')  # Changed from request.json to form data
        if not file_path:
            return jsonify({"error": "No file path provided"}), 400
        
        df = pd.read_csv(file_path)
        # Add your actual prediction pipeline logic here
        # ...
        # Preprocess the data
        processed_df = preprocess_data(df)
        
        # Make predictions
        predictions = model.predict(processed_df)
       
        class_mapping = {0: "Compensated Hypothyroid", 1: "Negative", 2: "Primary Hypothyroid",3:"Secondary Hypothyroid"}
        structured_results = [
            {"patient_id": idx + 1, "prediction": class_mapping[pred]}
            for idx, pred in enumerate(predictions)
        ]
        result_df = pd.DataFrame(structured_results)
        csv_filename = os.path.join(r"C:\Users\ys136\Desktop\Data Science\End to End ML Projects\thyroid_Prediction_System\artifacts\OUTPUT", "predictions.csv")
        result_df.to_csv(csv_filename, index=False)

        return jsonify({
            "message": "Custom file prediction successful",
            "csv_download_link": f"/download-predictions"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download-predictions', methods=['GET'])
def download_predictions():
    csv_filename = os.path.join(r"C:\Users\ys136\Desktop\Data Science\End to End ML Projects\thyroid_Prediction_System\artifacts\OUTPUT" ,"predictions.csv")
    
    # Debugging: Print the file path
    print(f"Checking for file: {csv_filename}")
    
    # Verify the file exists before sending
    if os.path.exists(csv_filename):
        return send_file(csv_filename, as_attachment=True, mimetype="text/csv")
    else:
        return jsonify({"error": "Predictions file not found"}), 404
    
@app.route('/predict-default', methods=['POST'])  # Changed to POST
def predict_default():
    try:
        # Define your default file path
        default_path = "path/to/default.csv"
        df = pd.read_csv(r"C:\Users\ys136\Desktop\Data Science\End to End ML Projects\thyroid_Prediction_System\Notebooks\data\data.csv")
        # Add your actual prediction pipeline logic here
        # ...
        processed_df = preprocess_data(df)
        
        # Make predictions
        predictions = model.predict(processed_df)
       
        class_mapping = {0: "Compensated Hypothyroid", 1: "Negative", 2: "Primary Hypothyroid",3:"Secondary Hypothyroid"}
        structured_results = [
            {"patient_id": idx + 1, "prediction": class_mapping[pred]}
            for idx, pred in enumerate(predictions)
        ]
        result_df = pd.DataFrame(structured_results)
        csv_filename = os.path.join(r"C:\Users\ys136\Desktop\Data Science\End to End ML Projects\thyroid_Prediction_System\artifacts\OUTPUT", "Custom_predictions.csv")
        result_df.to_csv(csv_filename, index=False)

        return jsonify({
            "message": "Custom file prediction successful",
            "csv_download_link": f"/download-predictions"
        })
        
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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