import joblib
import pandas as pd


# Loading Model n Scaler

loaded_model = joblib.load("Loan_MLP_Model.pkl")
loaded_scaler = joblib.load("Loan_Scaler.pkl")



# Testing New Data 


new_data = pd.DataFrame({

    'Age' : [25, 45, 32, 52, 69],
    'Income' : [420000, 850000, 650000, 1200000, 780000],
    'LoanAmount' : [250000, 350000, 600000, 450000, 300000],
    'CreditScore' : [720, 780, 650, 810, 735],
    'EmploymentYears' : [3, 12, 7, 25, 20],
    'ExistingLoans' : [1, 2, 3, 1, 2],
    'MonthlyDebt' : [15000, 22000, 28000, 18000, 25000],
    'LoanTerm' : [36, 48, 60, 36, 48],
    'PreviousDefault' : [0, 0, 1, 0, 1],
    'HomeOwnership' : [1, 0, 1, 2, 0]
})


# Scaling New Data 

new_scaled_data = loaded_scaler.transform(new_data)

new_pred = loaded_model.predict(new_scaled_data)

new_prob = loaded_model.predict_proba(new_scaled_data)

print("Prediction Probability : ",new_prob)


print("\nPredictions : ")

for i in range(len(new_pred)):
    if new_pred[i] == 1:
        result = 'High Risk'

    else:
        result = 'Low Risk'

    print("Applicant", i+1, result)

