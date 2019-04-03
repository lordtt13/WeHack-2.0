# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 00:16:35 2018

@author: Aryaman Sriram
"""
import pandas as pd


data = pd.read_csv("Hack_A_Thon_DataSet_Rev1.csv")

from xgboost import XGBRegressor
from scipy import stats
#author quality,duration,consistency in quality
#data = data.drop(['consistency in quality'],axis=1)
#data['quality_per_artifact'] = 1/(data["consistency in load"]*data["consistency in quality"])

#mask1 = stats.zscore(data['duration(days)'])<3
#data['duration(days)'][mask1]=0

#maskcomplex = data['complexity']>100
#data = data.drop(data[maskcomplex].index)

#masknoartifacts = data['#of artifacts']>800
#data = data.drop(data[masknoartifacts].index)

#maskperf = data['perf. as reviewer']>1250
#data = data.drop(data[maskperf].index)

#maskdur = data['duration(days)']>750
#data = data.drop(data[maskdur].index)
#data['prev_month_art'] = data['#of artifacts']-data["artifact trend"]
#data = data.drop(["complexity"], axis = 1)

x = data.drop(['total defects count'], axis = 1)
y = data["total defects count"]
#scaler = StandardScaler() 
#x['avg. quality of reviewer '] = scaler.fit_transform(np.array(x['avg. quality of reviewer '].reshape(-1,1)))
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
x_train,x_cv,y_train,y_cv = train_test_split(x_train,y_train,test_size=0.25,random_state=42)

from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso,LinearRegression,Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler,RobustScaler

rbs1 = RobustScaler()
rbs2 = RobustScaler()
x_train = rbs1.fit_transform(x_train)
x_test = rbs2.fit_transform(x_test)




pip_lasso = Pipeline([('kbest',SelectKBest()),('gb',GradientBoostingRegressor())])
param_grid = ({'kbest__k':[8]})
gbr = GridSearchCV(pip_lasso,param_grid).fit(x_train,y_train)
print(mean_squared_error(gbr.predict(x_cv),y_cv))
pred=gbr.predict(x_cv)
print(gbr.best_score_)
print(gbr.best_params_)
print(gbr.best_estimator_)
print(gbr.best_estimator_.named_steps["gb"].feature_importances_)


