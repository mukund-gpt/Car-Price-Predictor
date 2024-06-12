import streamlit as st
import pandas as pd
import numpy as np
import pypickle 

# Load the model and the dataset
# model = pickle.load(open("LinearRegression.pkl", 'rb'))
model=pypickle.load("LinearRegression.pkl",'rb')
car = pd.read_csv('Cleaned car.csv')

# Function to filter car models based on selected company
def get_car_models_by_company(selected_company):
    return car[car['company'] == selected_company]['name'].unique()

# Streamlit application
st.title('Car Price Prediction')

# UI Elements
companies = sorted(car['company'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = sorted(car['fuel_type'].unique())

# Select company
company = st.selectbox('Select Company', companies)

# Filter car models based on selected company
car_models = sorted(get_car_models_by_company(company))

# Select car model
car_model = st.selectbox('Select Car Model', car_models)

# Select other inputs
year = st.selectbox('Select Year', years)
fuel_type = st.selectbox('Select Fuel Type', fuel_types)
kms_driven = st.number_input('KMs Driven', min_value=0, step=1)

# Predict button
if st.button('Predict'):
    # Making prediction
    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], 
                                             columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    if prediction[0] < 0:
        prediction[0] = 0
    st.success(f'The predicted price of the car is ₹{np.round(prediction[0], 2)}')

# To run the app, use the command: streamlit run streamlit_app.py
