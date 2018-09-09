# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 11:06:54 2018

@author: tanma
"""


import numpy as np
import pandas as pd
import os
from xgboost import XGBRegressor

'''from gplearn.genetic import SymbolicRegressor

func = {'add','sub','mul','sin','cos','log','div'}
est_gp = SymbolicRegressor(population_size=300,
                           generations=100, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=42)

'''
data = pd.read_csv("C:\\Users\\Aryaman Sriram\\Documents\\WeHack_2.0-master/excel,csv,param,target sheet/Hack_A_Thon_DataSet_Rev1.csv")
#data['quality_per_artifact'] = 1/(data["consistency in load"]*data["consistency in quality"])
#mask = data['duration(days)']<=0
#data[mask]= np.mean(data['duration(days)'])
#data['prev_month_art'] = data['#of artifacts']-data["artifact trend"]
data = data.drop(["complexity"], axis = 1)

x = data.drop('total defects count', axis = 1)
y = data["total defects count"]

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
x_train,x_cv,y_train,y_cv = train_test_split(x_train,y_train,test_size=0.25,random_state=42)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso,LinearRegression,Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import r2_score

#my_model = XGBRegressor(learning_rate=0.05)

#my_model.fit(x_train,y_train)
#est_gp.fit(x_train,y_train)
#pred = est_gp.predict(x_cv)
#pred = my_model.predict(x_cv)

pip_lasso = Pipeline([('scaler',StandardScaler()),('kbest',SelectKBest()),('xgb',XGBRegressor())])
param_grid = ([{'kbest__k':[2,3,4,5,6,7,8]}])
gbr = GridSearchCV(pip_lasso,param_grid).fit(x_train,y_train)
print(mean_squared_error(gbr.predict(x_cv),y_cv))
print(gbr.best_score_)
print(gbr.best_params_)
