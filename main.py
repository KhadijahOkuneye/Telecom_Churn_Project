import streamlit as st
import joblib
import pandas as pd

st.title("Churn Prediction App")
model = joblib.load('gradient_boosting_final.pkl')
scaler = joblib.load('scaler_final.pkl')

cols = {
    'AccountWeeks': st.number_input('Account Weeks', min_value=0, value=100),       
  'ContractRenewal': st.number_input('Contract Renewal', min_value=0, value=1),
  'DataPlan' : st.number_input('Data PLan', min_value=0, value=0 ),           
  'DataUsage' : st.number_input('Data Usage', min_value=0.0, value=150.0),
'CustServCalls' :  st.number_input('Customer Service Calls', min_value=0, value=1),
'DayMins' :  st.number_input('Minutes Per Day', min_value=0.0, value=200.0),
'DayCalls' :  st.number_input('Calls Per Day', min_value=0, value=100),
'MonthlyCharge' :  st.number_input('Monthly Charge', min_value=0.0, value=60.0),
'OverageFee' :  st.number_input('Over Age Fee', min_value=0.0, value=10.0),
'RoamMins' :  st.number_input('Roam Mins', min_value=0.0, value=10.0),    
}

df = pd.DataFrame([cols])

if st.button('Predict', type='primary'):
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)
    st.success(f"Prediction: {prediction[0]}", icon="✅")
    if prediction[0] == 1:
       st.error("This customer is likely to CHURN!")
    else:
        st.info("This customer is likely to STAY!")
