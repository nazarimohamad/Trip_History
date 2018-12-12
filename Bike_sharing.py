# Bike sharing project/ kaggle competition

# Import lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('train.csv')

dataset.drop(['holiday', 'temp', 'registered', 'count'], axis = 1, inplace = True)
X = dataset.iloc[:, :7]
Y = dataset.iloc[:, 7]

# Import datetime and handle it
from datetime import datetime
X["hour"] = X.datetime.apply(lambda x : x.split()[1].split(":")[0])
X["year"] = X.datetime.apply(lambda x : x.split()[0].split('-')[0])
X["month"] = X.datetime.apply(lambda x : x.split()[0].split('-')[1])
X["day"] = X.datetime.apply(lambda x : x.split()[0].split('-')[2])

X.drop(['datetime'], axis = 1, inplace = True)

X['year'] = X['year'].map({'2012': 1, '2011': 0})

# Feature scaling for atem, humidity, windspeed
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X.iloc[:, 4:7] = sc.fit_transform(X.iloc[:, 4:7])

# Create dummy variables for categorical data and drop unnessecary data (dummy variable trap)
category = X[['season', 'workingday', 'weather', 'hour', 'year', 'month', 'day']]
category = pd.get_dummies(category)
category.drop(['hour_00', 'month_01', 'day_01'], axis = 1, inplace = True)

# modify X variable and add category variable to it
X.drop(['season', 'workingday', 'weather', 'hour', 'year', 'month', 'day'], axis = 1, inplace = True)
X = pd.concat([X, category], axis=1)
X.drop(['datetime'], axis = 1, inplace = True)

# Split to training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0) 

# Random Forest model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

# Score with RMSLE
score = np.sqrt(np.mean((np.log(1+y_pred[i]) - np.log(1+y_test[i]))**2))



