#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 19:17:16 2019

@author: nayani
"""

import sklearn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

#Question 1 
def load_train(Train_path ='train.xls', Test_path ='test.xls', gender_path ='gender_submission.xls'):
    csv_path= os.path.join(os.getcwd(), Train_path)
    csv_path_bis = os.path.join(os.getcwd(), Test_path)
    csv_path_ter = os.path.join(os.getcwd(), gender_path)
    dataset= pd.read_excel(csv_path).fillna(0)
    dataset_bis = pd.read_excel(csv_path_bis).fillna(0)
    dataset_ter = pd.read_excel(csv_path_ter).fillna(0)
#Question 2
    result = pd.merge(dataset_ter,dataset_bis, on='PassengerId')
    resultFinal = pd.concat([result,dataset])
    
    print('-------------------------DATASET Concat--------------------------------')
    print(result)
    print('--------------FIN---------------------------------------------------')
    
    return resultFinal
titanic = load_train()
titanic.head()
titanic.info()
#Question 3
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(titanic,test_size=0.2,random_state =400)


y_train = train_set['Survived']
X_train=train_set.drop(['Survived','Ticket','Cabin','Name','Sex','Embarked'],axis=1)
print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------------------')
y_test = test_set['Survived']
X_test=test_set.drop(['Survived','Ticket','Cabin','Name','Sex','Embarked'],axis=1)
#Question 4
from sklearn.metrics import mean_squared_error
from sklearn import svm

for kernel in ['linear','rbf'] :
    svmw = svm.SVC(kernel=kernel)
    svmw.fit(X_train, y_train)
    housing_predictions= svmw.predict(X_train)
    lin_mse = mean_squared_error(y_train, housing_predictions)
    print('Erreur (train), max depth=',kernel, ': ', lin_mse)
    housing_predictions= svmw.predict(X_test)
    lin_mse = mean_squared_error(y_test, housing_predictions)
    print('Erreur (test), max depth=',kernel, ': ', lin_mse)
plt.show()

