import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

def load_data():
    model = joblib.load('pklFiles\Qpistachio.pkl')
    scalar = joblib.load('pklFiles\Encode.pkl')
    return model, scalar

st.title('Pistachio Prediction')
st.write('This is a simple Pistachio prediction app')
area = st.number_input('Area', min_value=0.0, max_value=100000.0, value=0.0)
perimeter = st.number_input('Perimeter', min_value=0.0, max_value=100000.0, value=0.0)
major_axis = st.number_input('Major Axis', min_value=0.0, max_value=100000.0, value=0.0)
minor_axis = st.number_input('Minor Axis', min_value=0.0, max_value=100000.0, value=0.0)
eccentricity = st.number_input('Eccentricity', min_value=0.0, max_value=100000.0, value=0.0)
eqdiasq = st.number_input('EqdiaSq', min_value=0.0, max_value=100000.0, value=0.0)
solidity = st.number_input('Solidity', min_value=0.0, max_value=100000.0, value=0.0)
convex_area = st.number_input('Convex Area', min_value=0.0, max_value=100000.0, value=0.0)
extent = st.number_input('Extent', min_value=0.0, max_value=100000.0, value=0.0)
aspect_ratio = st.number_input('Aspect Ratio', min_value=0.0, max_value=100000.0, value=0.0)
roundness = st.number_input('Roundness', min_value=0.0, max_value=100000.0, value=0.0)
compactness = st.number_input('Compactness', min_value=0.0, max_value=100000.0, value=0.0)
shapefactor_1 = st.number_input('Shape Factor 1', min_value=0.0, max_value=100000.0, value=0.0)
shapefactor_2 = st.number_input('Shape Factor 2', min_value=0.0, max_value=100000.0, value=0.0)
shapefactor_3 = st.number_input('Shape Factor 3', min_value=0.0, max_value=100000.0, value=0.0)
shapefactor_4 = st.number_input('Shape Factor 4', min_value=0.0, max_value=100000.0, value=0.0)
button = st.button("Predict")

if button:
    model, scalar = load_data()
    prediction = scalar.inverse_transform(model.predict([[area, perimeter, major_axis, minor_axis, eccentricity, eqdiasq, solidity, convex_area, extent, aspect_ratio, roundness, compactness, shapefactor_1, shapefactor_2, shapefactor_3, shapefactor_4]]))
    st.write(f'The Pistachio is {prediction}')