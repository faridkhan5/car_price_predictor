from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd, numpy as np
import pickle

app = Flask(__name__)
CORS(app)

df = pd.read_csv('car_cleaned.csv')
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

@app.route('/')
def home():
    companies = sorted(df['company'].unique())
    car_models = sorted(df['name'].unique())
    years = sorted(df['year'].unique(), reverse=True)
    fuel_types = df['fuel_type'].unique()
    
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))

    user_inp = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    prediction = model.predict(user_inp)

    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run(debug=True)