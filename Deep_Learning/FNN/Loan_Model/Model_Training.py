import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# Loading Dataset 



data = pd.read_csv("Loan_Dataset_10K.csv")

print("First five records : ")
print(data.head())

print("Shape of Dataset : ")
print(data.shape)

print("Columns are : ")
print(data.columns)


# Missing Values 


print("missing Values in each Column is : ")
print(data.isnull().sum())

print("Missing Values in entire dataset : ")
print(data.isnull().sum().sum())



# Checking Target Class Balanced or not

target = 'Default'   

print("\nTarget Class Counts : ")
print(data[target].value_counts())

print("\nTarget Class Percentages : ")
print(data[target].value_counts(normalize=True) * 100)



# Encoding Categorical Data


data["PreviousDefault"] = data["PreviousDefault"].map({"Yes" : 1, "No" : 0})

data["HomeOwnership"] = data["HomeOwnership"].map({"Own" : 0, "Rent" : 1, "Mortgage" : 2})


# Separating X n Y 

X = data[['Age', 'Income', 'LoanAmount', 'CreditScore', 'EmploymentYears',
       'ExistingLoans', 'MonthlyDebt', 'LoanTerm', 'PreviousDefault',
       'HomeOwnership']]

Y = data['Default']



# Splitting Data --> Training n Testing


X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size= 0.3,
    random_state= 42,
    stratify=Y
)

print("Training Input Shape : ",X_train.shape)
print("Testing Input Shape : ",X_test.shape)
print("Training Output Shape : ",Y_train.shape)
print("Testing Output Shape : ",Y_test.shape)



# Feature Scaling 


Scaler = StandardScaler()

X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)


# Creating Model ---> MLP



model = MLPClassifier(
    hidden_layer_sizes=(128,64,32,16,8),
    activation='tanh',
    solver='adam',
    max_iter= 1000,
    random_state=42,
    learning_rate='adaptive',
    learning_rate_init=0.001
)


# Training Model 


model.fit(X_train, Y_train)

N_iterations = model.n_iter_
print("No of iterations Req For training : ",N_iterations)

final_loss = model.loss_
print("Final Loss : ",final_loss)



# Model Evaluation 



Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

Train_Accuracy = accuracy_score(Y_train, Y_pred_train)
Test_Accuracy = accuracy_score(Y_test, Y_pred_test)

print("Trainig Accuracy : ",Train_Accuracy*100,"%")
print("Testing Accuracy : ",Test_Accuracy*100,"%")

conf_matrix = confusion_matrix(Y_test, Y_pred_test)
print("Confusion Matrix : ",conf_matrix)


class_report = classification_report(Y_test, Y_pred_test)
print("Classification Report : ",class_report)


# Plotting Training Loss 


plt.plot(model.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("MLP Training Loss")
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()


# Preserving Model 



joblib.dump(model,"Loan_MLP_Model.pkl")
joblib.dump(Scaler,"Loan_Scaler.pkl")





