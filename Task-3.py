import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# load red wine data
data = pd.read_csv('winequality-red.csv', sep=';')
print(data.head())
print (data.shape)
print (data.describe())

# split data into training and test sets
y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# declare data preprocessing steps
X_train_scaled = preprocessing.scale(X_train)
#print (X_train_scaled)
pipeline = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(n_estimators=100))

# declare hyperparameters to tune
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                    'randomforestregressor__max_depth': [None, 5, 3, 1]}

# tune model using a cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)

# refit on the entire training set
print (clf.refit)

# evaluate model pipeline on test data
y_pred = clf.predict(X_test)
print (r2_score(y_test, y_pred))
print (mean_squared_error(y_test, y_pred))

# save model for future use
joblib.dump(clf, 'rf_regressor.pkl')
clf2 = joblib.load('rf_regressor.pkl')
clf2.predict(X_test)