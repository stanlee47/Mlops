import streamlit as st
import pandas as pd
import pickle as pk
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.title("Purchase Prediction App (Manual Input)")

# Input fields
group = st.selectbox("Group", ["Control", "Treatment"])
customer_segment = st.selectbox("Customer Segment", ["High Value", "Low Value", "Medium Value"])
sales_before = st.number_input("Sales Before", min_value=0.0, value=250.0)
sales_after = st.number_input("Sales After", min_value=0.0, value=300.0)
customer_satisfaction_before = st.number_input("Customer Satisfaction Before", min_value=0.0, max_value=100.0, value=75.0)
customer_satisfaction_after = st.number_input("Customer Satisfaction After", min_value=0.0, max_value=100.0, value=80.0)

# Prepare input as DataFrame
input_data = pd.DataFrame({
    'Group': [group],
    'Customer_Segment': [customer_segment],
    'Sales_Before': [sales_before],
    'Sales_After': [sales_after],
    'Customer_Satisfaction_Before': [customer_satisfaction_before],
    'Customer_Satisfaction_After': [customer_satisfaction_after]
})

st.subheader("Input Data Preview")
st.dataframe(input_data)

# Predict button
if st.button("Predict Purchase Made"):
    try:
        # Label Encoding
        label_encoder = LabelEncoder()
        for column in ['Group', 'Customer_Segment']:
            input_data[column] = label_encoder.fit_transform(input_data[column])

        # Standard Scaling
        scaler = StandardScaler()
        numerical_cols = ['Sales_Before', 'Sales_After', 'Customer_Satisfaction_Before', 'Customer_Satisfaction_After']
        input_data[numerical_cols] = scaler.fit_transform(input_data[numerical_cols])

        st.subheader("Preprocessed Data Preview")
        st.dataframe(input_data)

        # Load model
        model_file = 'models/model.pkl'
        with open(model_file, 'rb') as f:
            model = pk.load(f)

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = 'Yes' if prediction == 1 else 'No'

        st.success(f"Predicted Purchase Made: {result}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
