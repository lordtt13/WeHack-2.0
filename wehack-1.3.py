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
from keras.layers import Dense,Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2,l1
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dropout

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

sc = RobustScaler()
x_train = sc.fit_transform(x_train)

model = Sequential()
model.add(Dense(units = 128,input_dim = 8,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l2(0.01)))
model.add(Dense(units = 64,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l1(0.01)))
model.add(Dense(units = 32,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l1(0.01)))
model.add(Dense(units = 16,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l1(0.01)))
model.add(Dense(units = 8,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l1(0.01)))
model.add(Dense(units = 4,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l1(0.01)))
model.add(Dense(units = 2,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l1(0.01)))
model.add(Dense(units = 1,kernel_initializer = "glorot_uniform",activation = "relu",kernel_regularizer = l2(0.01)))
model.compile(loss = "mean_squared_error",optimizer = "nadam")

filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"
checkpoint = ModelCheckpoint(filepath,monitor = "loss", verbose = 1,save_best_only = True,mode = "min")
callbacks_list = [checkpoint]
    
model.fit(x_train,y_train,epochs = 100,batch_size = 1,callbacks = callbacks_list)


preds = np.rint(model.predict(sc.transform(x_test)))
for i in range(len(preds)-1):
    if preds[i]<0:
        preds[i] = 0

print(mean_squared_error(preds,y_test))




