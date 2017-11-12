#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:02:43 2017

@author: josef
"""

#importando librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importando el dataset o los datos en archivo csv
dataset = pd.read_csv('50_Startups.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#encoding the indepent variables 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#avoiding the Dummy variable train
x = x[:, 1:] 

#spliting the dataset into the Training set and Test set 
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Fitting multiple Linear Regression to the Training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test set  results 
y_pred = regressor.predict(x_test)


