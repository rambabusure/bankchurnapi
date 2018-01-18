# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:16:51 2018

@author: RBabu
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import kfoldvalidation as kv
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC



def modelevaluator(datafilename):
    X,y =  formatdata(datafilename)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    model_accuracy_values = {}
    #"Logistic Regression:"
    model_accuracy_values["LR"] = kv.accuracy(y, kv.run_cv(X,y,LR))
    #"Gradient Boosting Classifier"
    model_accuracy_values["GBC"] =   kv.accuracy(y, kv.run_cv(X,y,GBC))
    # "Support vector machines:"
    model_accuracy_values["SVC"]= kv.accuracy(y, kv.run_cv(X,y,SVC))
    # "random forest:"
    model_accuracy_values["RF"] =  kv.accuracy(y, kv.run_cv(X,y,RF))
    #k-nearest-neighbors:
    model_accuracy_values["KNN"] =  kv.accuracy(y, kv.run_cv(X,y,KNN))
    return max(model_accuracy_values, key=lambda k: model_accuracy_values[k]) 


def formatdata(datafilename):
    churn_df = pd.read_csv(datafilename)
    churn_result = churn_df['Churn?']
    y= np.where(churn_result =='True.',1,0)
    # We don't need these columns
    to_drop = ['State','Area Code','Phone','Churn?']
    churn_feat_space =  churn_df.drop(to_drop, axis=1)
    # 'yes'/'no' has to be converted to boolean values
    # NumPy converts these from boolean to 1. and 0. later
    yes_no_cols = ["Int'l Plan","VMail Plan"]
    churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'
    X = churn_feat_space.as_matrix().astype(np.float)
    return (X,y)


def formatinputdata(datafilename):
    churn_df = pd.read_csv(datafilename)
   
    # We don't need these columns
    to_drop = ['State','Area Code','Phone','Churn?']
    churn_feat_space =  churn_df.drop(to_drop, axis=1)
    # 'yes'/'no' has to be converted to boolean values
    # NumPy converts these from boolean to 1. and 0. later
    yes_no_cols = ["Int'l Plan","VMail Plan"]
    churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'
    X = churn_feat_space.as_matrix().astype(np.float)
    return X

accuratemodel = modelevaluator('churn.csv')
print(accuratemodel)
X,y = formatdata('churn.csv')
kv.generateandpersistmodel(X,y,GBC)
X_Input,y = formatinputdata('churn_predict.csv')
y_pred= kv.predictresults(X_Input.reshape(1,-1))
print(y_pred)