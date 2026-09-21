import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input


import matplotlib.pyplot as plt

import joblib


# Data Loading


data = pd.read_csv('Churn_Modelling.csv')

print(data.head())

print("Dataset Shape:")
print(data.shape)

print("Dataset Info:")
print(data.info())

print("Duplicated Rows:")
print(data.duplicated().sum())


# Data Distribution


print("Exited Distribution:")
print(data['Exited'].value_counts())

print("Gender Distribution:")
print(data['Gender'].value_counts())

print("Geography Distribution:")
print(data['Geography'].value_counts())


# Removing Unnecessary Columns


data.drop(
    columns=['RowNumber', 'CustomerId', 'Surname'],
    inplace=True
)


# Encoding Categorical Columns


data = pd.get_dummies(
    data,
    columns=['Geography', 'Gender'],
    drop_first=True
)

print("Encoded Dataset:")
print(data)


# Separating X and Y


x = data.drop(columns=['Exited'])
y = data['Exited']


# Train-Test Split


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=1
)

print("X Train Shape:", x_train.shape)
print("X Test Shape:", x_test.shape)
print("Y Train Shape:", y_train.shape)
print("Y Test Shape:", y_test.shape)


# Feature Scaling


scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# FNN Model Creation


model = Sequential([
    Input(shape=(x_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

print(model.summary())


# Model Compilation


model.compile(
    loss='binary_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)


# Model Training


history = model.fit(
    x_train_scaled,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)


# Prediction


y_log = model.predict(x_test_scaled)

y_pred = np.where(y_log > 0.5, 1, 0)



# Model Evaluation


testing_accuracy = accuracy_score(y_test, y_pred)
print("Testing Accuracy:", testing_accuracy * 100, "%")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# Training Graphs


fig, ax = plt.subplots(
    2,
    1,
    figsize=(10, 8)
)



# Loss Graph


ax[0].plot(
    history.history['loss'],
    label='Training Loss'
)

ax[0].plot(
    history.history['val_loss'],
    label='Validation Loss'
)

ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')

ax[0].set_title(
    'Training and Validation Loss'
)

ax[0].legend()



# Accuracy Graph


ax[1].plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

ax[1].plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')

ax[1].set_title(
    'Training and Validation Accuracy'
)

ax[1].legend()


# Adjust layout
plt.tight_layout()


# Save graphs
plt.savefig(
    'training_validation_graphs.png',
    dpi=300,
    bbox_inches='tight'
)


# Display graphs

plt.show(block=True)


# Saving FNN model and scaler


model.save('credit_card_FNN_model.keras')

joblib.dump(scaler, 'credit_card_scaler.pkl')