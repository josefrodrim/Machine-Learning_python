#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:16:03 2017

@author: josef
"""

#Simple linear Regression

#importando librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importando el dataset o los datos en archivo csv
dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#spliting the dataset into the Training set and Test set 
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#Fitting Sample Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test set results  
y_pred = regressor.predict(x_test)

#viasualising the Training Set Results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salario vs Experiencia')
plt.xlabel('Años de experiencia')
plt.ylabel('Salario')
plt.show()

#viasualising the Test Set Results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salario vs Experiencia')
plt.xlabel('Años de experiencia')
plt.ylabel('Salario')
plt.show()