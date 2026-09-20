import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset Loading

data = pd.read_csv("Employee_Attrition.csv")

print("Fisrt five rows : ")
print(data.head())

print("Shape of Dataset : ")
print(data.shape)

print("Columns are : ")
print(data.columns)


# Missing Values 


print("missing values in each column is : ")
print(data.isnull().sum())

print("missing values in entire dataset are :")
print(data.isnull().sum().sum())


# Target class Balanced or not 

target = 'Attrition'

print("\nTarget class counts : ")
print(data[target].value_counts())

print("\nTarget Class Percentages : ")
print(data[target].value_counts(normalize=True) * 100)


# Encoding categorical data 

data["OverTime"] = data["OverTime"].map({"Yes" : 1, "No" : 0})

data["Attrition"] = data["Attrition"].map({"Yes" : 1, "No" : 0})


# Separating x and y

x = data[['Age', 'MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears',
       'DistanceFromHome', 'JobSatisfaction', 'WorkLifeBalance', 'OverTime',
       'NumCompaniesWorked', 'TrainingTimesLastYear']]

y = data['Attrition']


# Splitting data for training and testing 

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size= 0.3,
    random_state= 42,
    stratify= y
)


print("Training input shape : ",x_train.shape)
print("Testing input shape : ",x_test.shape)
print("Training output shape : ",y_train.shape)
print("Testing output shape : ",y_test.shape)


# Feature scaling 


scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



# Model creation --> MLP 

model = MLPClassifier(
    hidden_layer_sizes=(88,66,44,22,11,2),
    activation='relu',
    solver='adam',
    max_iter= 1000,
    random_state= 42,
    learning_rate='adaptive',
    learning_rate_init= 0.001,
    batch_size= 256
    
)


# Training model 


model.fit(x_train, y_train)

n_iterations = model.n_iter_
print("No. of iterations required for training : ",n_iterations)

final_loss = model.loss_
print("final loss : ",final_loss)



# Model evaluation 

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)


train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)


print("Trainig Accuracy : ",train_accuracy*100,"%")
print("Testing Accuracy : ",test_accuracy*100,"%")


conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion matrix : ",conf_matrix)

class_report = classification_report(y_test, y_pred_test)
print("Classification report : ",class_report)


# Plotting Training loss 

plt.plot(model.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("MLP Training Loss")
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()


# model preserving 

joblib.dump(model,"employee_mlp_model.pkl")
joblib.dump(scaler,"employee_scaler.pkl")