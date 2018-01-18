# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:09:01 2018

@author: RBabu
"""
import numpy as np
from sklearn.cross_validation import KFold




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