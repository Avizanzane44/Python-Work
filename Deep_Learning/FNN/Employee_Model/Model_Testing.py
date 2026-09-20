import joblib
import pandas as pd

# Loading Saved model and scaler


loaded_model = joblib.load("employee_mlp_model.pkl")
loaded_scaler = joblib.load("employee_scaler.pkl")



# Testing New Data

new_data = pd.DataFrame({
    'Age': [22, 28, 35, 41, 48, 26, 33, 56, 39, 30],
    
    'MonthlyIncome': [
        25000, 42000, 68000, 95000, 145000,
        32000, 75000, 160000, 58000, 45000
    ],
    
    'YearsAtCompany': [
        1, 2, 7, 12, 18,
        1, 4, 22, 9, 3
    ],
    
    'TotalWorkingYears': [
        2, 5, 10, 17, 25,
        3, 8, 30, 14, 6
    ],
    
    'DistanceFromHome': [
        35, 18, 7, 3, 2,
        45, 12, 28, 5, 22
    ],
    
    'JobSatisfaction': [
        1, 2, 3, 4, 2,
        1, 4, 3, 2, 4
    ],
    
    'WorkLifeBalance': [
        1, 2, 3, 4, 2,
        1, 4, 3, 2, 4
    ],
    
    'OverTime': [
        1, 1, 0, 0, 1,
        1, 0, 0, 1, 0
    ],
    
    'NumCompaniesWorked': [
        4, 3, 2, 1, 2,
        5, 1, 3, 4, 2
    ],
    
    'TrainingTimesLastYear': [
        2, 3, 4, 5, 3,
        1, 6, 4, 2, 5
    ]
})


# Scaling new data

New_scaled_data = loaded_scaler.transform(new_data)

new_pred = loaded_model.predict(New_scaled_data)

new_prob = loaded_model.predict_proba(New_scaled_data)

print("Prediction Probablity ",new_prob)


print("\nPredicted Attrition:")

for i in range(len(new_pred)):

    if new_pred[i] == 1:
        result = "Yes"
    else:
        result = "No"

    print(f"Employee {i+1}: {result}")