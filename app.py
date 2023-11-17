import streamlit as st
import pandas as pd
import pickle
import joblib


#loading the scaler
scaler=joblib.load('C:\\Users\\hp\\Desktop\\Year 3 Sem 2\\Introduction to AI\\65542024_Churning_Customers\\scaler.pkl')


#loading the Random Forest Regressor model
model=joblib.load('C:\\Users\\hp\\Desktop\\Year 3 Sem 2\\Introduction to AI\\65542024_Churning_Customers\\model.pkl')

#loading the list of features used to train the model
with open('C:\\Users\\hp\\Desktop\\Year 3 Sem 2\\Introduction to AI\\65542024_Churning_Customers\\top_features.pkl','rb') as file:
    top_features=pickle.load(file)


def get_user_inputs():
    user_inputs = {}
    for feature in top_features:
        user_inputs[feature] = st.sidebar.number_input(f"Enter value for {feature}", value=0.0)
    return user_inputs

def make_prediction(scaled_inputs):
    prediction = model.predict(scaled_inputs)
    prediction_proba= model.predict_proba(scaled_inputs)
    return prediction, prediction_proba


st.title('Customer Churn Predictor')
st.sidebar.header('Enter Feature Values')


# Get user inputs
user_inputs = get_user_inputs()

# Preprocess user inputs using the scaler
input_data = pd.DataFrame([user_inputs])
scaled_inputs = scaler.transform(input_data)

if st.sidebar.button('Predict'):
    prediction, prediction_proba = make_prediction(scaled_inputs)


    # Display prediction and confidence level
    st.write("## Prediction")
    if prediction[0] == 1:
        st.write("Customer is likely to churn.")
    else:
        st.write("Customer is likely to stay.")

    st.write("## Confidence Level")
    #st.write(f"The model is {prediction_proba[0][1] * 100:.2f}% confident about this prediction.")
