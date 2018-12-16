# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 14:20:05 2018

@author: tanma
"""


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso,LinearRegression,TheilSenRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor,BernoulliRBM
from mlxtend.regressor import StackingRegressor
from xgboost import XGBRegressor



cwd = os.getcwd()
print (cwd)

data = pd.read_csv("Hack_A_Thon_DataSet_Rev1.csv")
data['quality_per_artifact'] = 1/(data["consistency in load"]*data["consistency in quality"])
mask = data['duration(days)']<=0
data = data.drop(data[mask].index)
data = data.drop(["complexity","perf. as reviewer"], axis = 1)

'''from gplearn.genetic import SymbolicRegressor

func = {'add','sub','mul','sin','cos','log','div'}
est_gp = SymbolicRegressor(population_size=300,
                           generations=100, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=42)

'''

x = data.drop('total defects count', axis = 1)
y = data["total defects count"]

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)



pip_lasso = Pipeline([('scaler',StandardScaler()),('kbest',SelectKBest()),('gbr',XGBRegressor())])
param_grid = ([{'kbest__k':[2,3,4,5,6,7],'gbr__learning_rate':[0.05,0.1,0.2,0.3,0.4,0.5]}])
gbr = GridSearchCV(pip_lasso,param_grid).fit(x_train,y_train)
preds = np.rint(gbr.predict(x_test))
for i in range(len(preds)-1):
    if preds[i]<0:
        preds[i] = 0

print(mean_squared_error(preds,y_test))
print(gbr.best_score_)
print(gbr.best_params_)


print(mean_squared_error(gbr.predict(x_test),y_test))
print(gbr.score(x_test,y_test))
lasso = Lasso()
ridge = TheilSenRegressor()
rf = GradientBoostingRegressor()
stack = StackingRegressor(regressors=(lasso, ridge),
                            meta_regressor=rf, 
                            use_features_in_secondary=True)
pipeline = make_pipeline(StandardScaler(), stack)
params = {'stackingregressor__lasso__alpha': [0.1, 1.0, 10.0]}
grid = GridSearchCV(
    verbose=1,
    estimator=pipeline, 
    param_grid=params, 
    cv=5,
    refit=True
)
grid.fit(x_train, y_train)
print("Best: %f using %s" % (grid.best_score_, grid.best_params_))

"""def neural_net_model(X_data,input_dim):
    W_1 = tf.Variable(tf.random_uniform([input_dim,10]))
    b_1 = tf.Variable(tf.zeros([10]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)

    # layer 1 multiplying and adding bias then activation function
    W_2 = tf.Variable(tf.random_uniform([10,10]))
    b_2 = tf.Variable(tf.zeros([10]))
    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)
    # layer 2 multiplying and adding bias then activation function
    W_O = tf.Variable(tf.random_uniform([10,1]))
    b_O = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_2,W_O), b_O)
    # O/p layer multiplying and adding bias then activation function
    # notice output layer has one node only since performing #regression
    return output"""
