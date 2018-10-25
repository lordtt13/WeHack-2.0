# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 21:27:43 2018

@author: tanma
"""

import numpy as np
import pandas as pd

data = pd.read_csv("excel,csv,param,target sheet\Hack_A_Thon_DataSet_Rev1.csv")

from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA,KernelPCA
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score,cross_val_predict
from xgboost import XGBRegressor

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2,l1
from keras.wrappers.scikit_learn import KerasRegressor

data = pd.read_csv("Hack_A_Thon_DataSet_Rev1.csv")
data['quality_per_artifact'] = 1/(data["consistency in load"]*data["consistency in quality"])
mask = data['duration(days)']<=0
data = data.drop(data[mask].index)
data = data.drop(["complexity","perf. as reviewer"], axis = 1)

x = data.drop(['total defects count'], axis = 1)
y = data["total defects count"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
x_train,x_cv,y_train,y_cv = train_test_split(x_train,y_train,test_size=0.25,random_state=42)

def baseline_model():
    model = Sequential()
    model.add(Dense(units = 8,input_dim = 8,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l2(0.01)))
    model.add(Dense(units = 8,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l1(0.01)))
    model.add(Dense(units = 8,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l1(0.01)))
    model.add(Dense(units = 8,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l1(0.01)))
    model.add(Dense(units = 8,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l1(0.01)))
    model.add(Dense(units = 8,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l1(0.01)))
    model.add(Dense(units = 8,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l1(0.01)))
    model.add(Dense(units = 1,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l1(0.01)))
    model.compile(loss = "mean_squared_error",optimizer = "nadam")
    return model


estimator = KerasRegressor(build_fn = baseline_model,epochs = 50,batch_size = 50,verbose = 0)
pip_lasso = Pipeline([('scaler',RobustScaler()),('model',estimator)])
gbr = pip_lasso.fit(x_train,y_train)

preds = np.rint(gbr.predict(x_cv))
for i in range(len(preds)-1):
    if preds[i]<0:
        preds[i] = 0

print(mean_squared_error(preds,y_cv))





