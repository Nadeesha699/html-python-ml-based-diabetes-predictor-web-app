from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)
CORS(app)
with open('./model/trained_model.pkl','rb') as model_file:
    model = pickle.load(model_file)

with open('./model/scaler.pkl','rb') as scaler_file:
    scaler = pickle.load(scaler_file)    


@app.route('/api/predict-diabetes',methods=['POST'])
def predict():
    data = request.get_json()
    input_features = [
        data['Pregnancies'],
        data['Glucose'],
        data['BloodPressure'],
        data['SkinThickness'],
        data['Insulin'],
        data['BMI'],
        data['DiabetesPedigreeFunction'],
        data['Age'],
    ]
    feature_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    input_df = pd.DataFrame([input_features],columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        results = "Diabetic"
    else:
        results = "Non-Diabetic"  
    return jsonify ({'result': results })       
    

if __name__ == '__main__':
    app.run(debug=True)