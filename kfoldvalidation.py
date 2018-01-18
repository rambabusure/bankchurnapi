# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:09:01 2018

@author: RBabu
"""
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.externals import joblib




def run_cv(X,y,clf_class, **kwargs):
    kf = KFold(len(y),n_folds=3, shuffle=True)
    y_pred = y.copy()
   
    for train_index,test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train=  y[train_index]
        
        clf =clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index]= clf.predict(X_test)
    return y_pred

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# this method will run and presist the model based on the high accuracy model
def generateandpersistmodel(X,y,clf_class):
    kf = KFold(len(y),n_folds=3, shuffle=True)
    for train_index,test_index in kf:
        X_train = X[train_index]
        y_train=  y[train_index]
        clf = clf_class()
        clf.fit(X_train,y_train)
        joblib.dump(clf,'bankchurnmodel.pkl')   
    return

#  this method will load the persisted model and return the predicted results.
def predictresults(inputdata):
    clf = joblib.load('bankchurnmodel.pkl')
    y_pred=  clf.predict(inputdata)
    return y_pred
    
    