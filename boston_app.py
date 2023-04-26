import streamlit as st
import pandas as pd
import numpy as np
from description import Feature_description, load_data
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

st.header(":blue[Boston House Price Prediction]")
st.subheader("This app will predict the median value of owner-occupied homes in $1000s (MEDV) of the Boston house dataset.")
 
#loading the data
data = pd.read_csv(r'C:\Users\MASTER\Boston\boston.csv')

#adding feature description to sidebar
st.sidebar.subheader("**:blue[Check features description here.]**")
st.sidebar.write('Select feature')
feature = st.sidebar.selectbox('Select feature', options = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'LSTAT'], label_visibility = 'collapsed' )
if st.sidebar.button('Show feature description'):
    Feature_description(feature)
    
#setting input parameters
left_column, middle_column, right_column = st.columns(3)

CRIM = left_column.number_input('CRIM', min_value = 0.000, max_value = 100.000, step = 0.001)

INDUS = middle_column.number_input('INDUS', min_value = 0.000, max_value = 30.000, step = 0.001)

NOX = right_column.number_input('NOX', min_value = 0.000, max_value = 1.000, step = 0.001)

RM = left_column.number_input('RM', min_value = 3.000, max_value = 9.000, step = 0.001)

AGE = middle_column.number_input('AGE', min_value = 0.000, max_value = 100.000, step = 0.001)

DIS = right_column.number_input('DIS', min_value = 0.000, max_value = 13.000, step = 0.001)

TAX = left_column.number_input('TAX', min_value = 187.0, max_value = 711.0, step = 0.1)

PTRATIO = middle_column.number_input('PTRATIO', min_value = 12.00, max_value = 22.00, step = 0.01)

LSTAT = right_column.number_input('LSTAT', min_value = 0.00, max_value = 40.00, step = 0.01)


#loading the model
with open('price_predictor.sav', 'rb') as m:
    model = pickle.load(m)
    
#creating a dataframe of input features to make it 2-dimensional
input_features = {"CRIM" : CRIM, "INDUS" : INDUS, "NOX": NOX, "RM": RM, "AGE":AGE, "DIS": DIS, "TAX": TAX, "PTRATIO": PTRATIO, "LSTAT": LSTAT}
input_features_frame = pd.DataFrame(input_features, index = [0])


#feature scaling
scaler = MinMaxScaler()
input_scaled = scaler.fit_transform(input_features_frame)
#input_feature = input_scaled.reshape(-1, 1)

st.write(":blue[Summary of feature specified]")
st.write(input_features_frame)

#predicting 
predicted_price = model.predict(input_scaled)

#printing prediction
if st.button('Show predicted price'):
    st.subheader(f":blue[The predicted price of the house is between USD {np.round((predicted_price[0] - 2.8) * 1000, )} and USD {np.round((predicted_price[0] + 2.8) * 1000)}]")





