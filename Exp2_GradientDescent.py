# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:23:18 2019

@author: Dipak Kumar
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 13:36:22 2019

@author: DIPAK

Logistic Regression - Coursera 
"""
import pandas as pd
from matplotlib import pyplot as plt
import math
import numpy as np
from scipy.optimize import fmin


#file = input("input file location: \t")
file = "C:\ex2.txt"

#data = pd.read_txt()
data = pd.read_csv(file, sep=",", header=None)
data.columns = ["x1", "x2", "y"]

features = ['x1','x2']

X = data[features]
#print(X.describe())
y = data['y']
#print(y.describe())
admitted = data.loc[y == 1]
# filter out the applicants that din't get admission
not_admitted = data.loc[y == 0]
# plots
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1],s=10,marker = '^', label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
plt.legend()
plt.show()

x1 = np.ones(len(y))
#np.append(X,x1,axis = 1)
X_new = np.column_stack((X,x1))
X = X_new

temp = X[:,0].copy()
X[:,0] = X[:,2]
X[:,2] = temp

temp = X[:,1].copy()
X[:,1] = X[:,2]
X[:,2] = temp

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

in_theta = np.zeros(3)

def costfunction(theta,X,y):
    m = len(y)
    J = 0 
    
    error = sigmoid(np.dot(X,theta))
    on = np.ones_like(y)
    J = np.sum((-y*np.log(error))-((on-y)*np.log(on-error)))/m
    
    return J
   
def gradient(theta,X,y,iterations,alpha):
    m = len(y)
    cost = [0.0 for i in range(iterations)]
    for i in range(iterations):
        error = sigmoid(np.dot(X,theta))
        theta = theta - alpha*(1/m)*(X.T.dot(error-y))
        cost[i] = costfunction(theta,X,y)
    
    return theta, cost


grad = np.zeros(len(in_theta))

J = costfunction(in_theta,X,y)

test_theta = np.array([-24,0.20,0.20])

J = costfunction(test_theta,X,y)
iterations = 1500
theta,cost = gradient(in_theta,X,y,iterations,0.001)


x1 = float(input("Enter marks 1"))
x2 = float(input("Enter marks 2"))
X_predict = np.array([1,x1,x2])
predict = theta[0] + (theta[1]*x1) + (theta[2]*x2)