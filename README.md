# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2.Load the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.

 
 
 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: M.Dhanush
RegisterNumber:  25009955

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Placement_Data.csv')
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
Y

theta = np.random.randn(X.shape[1])
y =Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h= sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta

theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations = 1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred

y_pred = predict(theta,X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(y_pred)

print(Y)

xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
*/
```

## Output:
![logistic regression using gradient descent](sam.png)
<img width="1228" height="448" alt="Screenshot 2025-12-11 091308" src="https://github.com/user-attachments/assets/9f1c9a2b-c518-426d-9b28-8971eac9cdb4" />

<img width="355" height="312" alt="Screenshot 2025-12-11 091354" src="https://github.com/user-attachments/assets/3184a1fb-4d2e-43be-9890-ece2144dd019" />

<img width="1011" height="425" alt="Screenshot 2025-12-11 091403" src="https://github.com/user-attachments/assets/effd7490-1146-49cf-9e0f-fcb1797e2e7f" />


<img width="758" height="213" alt="Screenshot 2025-12-11 091413" src="https://github.com/user-attachments/assets/75436c40-4973-4a7f-ab3e-2fab63c611ba" />

<img width="803" height="138" alt="Screenshot 2025-12-11 091424" src="https://github.com/user-attachments/assets/68e4d2ed-9e70-4a82-8dc1-72af27b69405" />

<img width="748" height="141" alt="Screenshot 2025-12-11 091432" src="https://github.com/user-attachments/assets/b9cb026d-d8a5-4b22-9679-d1969f0ba7c7" />

<img width="55" height="32" alt="Screenshot 2025-12-11 091448" src="https://github.com/user-attachments/assets/0a62b3e3-84e5-4ceb-8e78-12e24600e77c" />


<img width="35" height="17" alt="Screenshot 2025-12-11 091454" src="https://github.com/user-attachments/assets/5d8002e7-7465-43df-9ebe-cb62ecf73fa6" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

