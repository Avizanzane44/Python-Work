import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model



# Load Trained Model and Scaler



model = load_model('credit_card_FNN_model.keras')

scaler = joblib.load('credit_card_scaler.pkl')



# Testing Data


testing_data = pd.DataFrame([
    {
        'CreditScore': 650,
        'Age': 35,
        'Tenure': 5,
        'Balance': 75000,
        'NumOfProducts': 1,
        'HasCrCard': 1,
        'IsActiveMember': 1,
        'EstimatedSalary': 85000,
        'Geography_Germany': False,
        'Geography_Spain': False,
        'Gender_Male': True
    },

    {
        'CreditScore': 450,
        'Age': 55,
        'Tenure': 2,
        'Balance': 120000,
        'NumOfProducts': 1,
        'HasCrCard': 1,
        'IsActiveMember': 0,
        'EstimatedSalary': 65000,
        'Geography_Germany': True,
        'Geography_Spain': False,
        'Gender_Male': False
    },

    {
        'CreditScore': 720,
        'Age': 28,
        'Tenure': 8,
        'Balance': 0,
        'NumOfProducts': 2,
        'HasCrCard': 1,
        'IsActiveMember': 1,
        'EstimatedSalary': 95000,
        'Geography_Germany': False,
        'Geography_Spain': True,
        'Gender_Male': True
    },

    {
        'CreditScore': 580,
        'Age': 48,
        'Tenure': 3,
        'Balance': 95000,
        'NumOfProducts': 1,
        'HasCrCard': 0,
        'IsActiveMember': 0,
        'EstimatedSalary': 55000,
        'Geography_Germany': True,
        'Geography_Spain': False,
        'Gender_Male': False
    }
])


# Display Testing Data


print("Testing Data:")
print(testing_data)



# Feature Scaling


testing_data_scaled = scaler.transform(testing_data)


# Prediction


y_probability = model.predict(testing_data_scaled)

y_prediction = np.where(
    y_probability > 0.5,1,0)


# Display Results


print("\nPrediction Results:")

for i in range(len(testing_data)):

    probability = y_probability[i][0]
    prediction = y_prediction[i]

    print("\nCustomer", i + 1)
    print("Churn Probability:", probability)

    if prediction == 1:
        print("Prediction: Customer is likely to churn")
    else:
        print("Prediction: Customer is likely to stay")
