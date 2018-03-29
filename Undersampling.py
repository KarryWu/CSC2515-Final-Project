#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 00:43:34 2017

@author: wukairui
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import roc_curve,auc
from sklearn.ensemble import RandomForestClassifier  
from sklearn import tree  
from sklearn import neighbors  
from sklearn.naive_bayes import GaussianNB 
import matplotlib.pyplot as plt



def load_clean_data():
    data=pd.read_csv('cs-training.csv', sep=',',header = 0)
    data_withoutna = data.dropna(axis = 0, how = "any")
    data_withoutna = np.array(data_withoutna.iloc[:,1:])
    new_data1 = data_withoutna[(np.abs(stats.zscore(data_withoutna[:,1])) < 3)]   ##remove NA
    new_data2 = new_data1[(new_data1[:,2] < 90) & (new_data1[:,2] > 20)]      ##remove outlier
    new_data3 = new_data2[new_data2[:,3] < 50]                                  ##remove outlier
    new_data4 = new_data3[(new_data3[:,4] < 1) & (new_data3[:,4] > 0)]          ##remove outlier
    new_data5 = new_data4[(np.abs(stats.zscore(new_data4[:,5])) < 3)]           ##remove outlier
    data_set = new_data5[:,1:]
    data_label = new_data5[:,0]
    X_train, X_test, y_train, y_test = train_test_split(data_set, data_label, test_size=0.25, random_state = 0)
    return X_train, X_test, y_train, y_test

##logistic regression
def logreg(X_train, y_train, X_test, y_test):
    logreg = linear_model.LogisticRegression()  
    logreg.fit(X_train, y_train)  
    predictions=logreg.predict_proba(X_test)
    false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:
        , 1])
    roc_auc=auc(false_positive_rate,recall)
    return roc_auc

##random forest
def rf(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier()  
    rf.fit(X_train,y_train)
    predictions=rf.predict_proba(X_test)
    false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:
        , 1])
    roc_auc=auc(false_positive_rate,recall)
    return roc_auc

##Decision Tree
def decitree(X_train, y_train, X_test, y_test):
    clf = tree.DecisionTreeClassifier(criterion='entropy')  
    clf.fit(X_train, y_train) 
    predictions=clf.predict_proba(X_test)
    false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:
        , 1])
    roc_auc=auc(false_positive_rate,recall)
    return roc_auc

##KNN
def knn(X_train, y_train, X_test, y_test):
    knn = neighbors.KNeighborsClassifier()  
    knn.fit(X_train, y_train) 
    predictions=knn.predict_proba(X_test)
    false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:
        , 1])
    roc_auc=auc(false_positive_rate,recall)
    return roc_auc 

##Naive Bayes
def nb(X_train, y_train, X_test, y_test):
    nb = GaussianNB()    
    nb.fit(X_train, y_train)   
    predictions=nb.predict_proba(X_test)
    false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:
        , 1])
    roc_auc=auc(false_positive_rate,recall)
    return roc_auc 



def undersampling(N,X_train,y_train):
    aimed_X_train = X_train[np.argwhere(y_train == 0).squeeze()]
    index = np.random.choice(len(aimed_X_train),round(len(aimed_X_train)*(1-N)),replace = False)
    rest_data = aimed_X_train[index]
    new_X_train = np.vstack((X_train[np.argwhere(y_train == 1).squeeze()], rest_data))
    new_y_train = np.vstack((np.ones(((len(X_train)-len(aimed_X_train)),1)),np.zeros((len(rest_data),1))))
    new_dataset = np.hstack((new_X_train,new_y_train))
    np.random.shuffle(new_dataset)
    return new_dataset

X_train, X_test, y_train, y_test = load_clean_data()

##calculate the original AUC
original_X_train = X_train
original_y_train = y_train
AUC_matrix = np.zeros((5,1))
AUC_matrix[0,0] = logreg(original_X_train, original_y_train, X_test, y_test)
AUC_matrix[1,0] = rf(original_X_train, original_y_train, X_test, y_test)
AUC_matrix[2,0] = decitree(original_X_train, original_y_train, X_test, y_test)
AUC_matrix[3,0] = knn(original_X_train, original_y_train, X_test, y_test)
AUC_matrix[4,0] = nb(original_X_train, original_y_train, X_test, y_test)
ratio = np.zeros((1,1))
ratio[0,0] = sum(original_y_train)/len(original_y_train)

##calculate the AUC under different label 1 ratio
indicator = ratio[0,0]
while indicator < 0.5:
    new_dataset = undersampling(0.1, X_train, y_train)
    new_X_train = new_dataset[:,:10]
    new_y_train = new_dataset[:,10]
    new_AUC_matrix = np.zeros((5,1))
    new_ratio = np.zeros((1,1))
    new_AUC_matrix[0,0] = logreg(new_X_train, new_y_train, X_test, y_test)
    new_AUC_matrix[1,0] = rf(new_X_train, new_y_train, X_test, y_test)
    new_AUC_matrix[2,0] = decitree(new_X_train, new_y_train, X_test, y_test)
    new_AUC_matrix[3,0] = knn(new_X_train, new_y_train, X_test, y_test)
    new_AUC_matrix[4,0] = nb(new_X_train, new_y_train, X_test, y_test)
    new_ratio[0,0] = sum(new_y_train)/len(new_y_train)
    indicator = new_ratio[0,0]
    print("indicator =",indicator)
    ratio = np.hstack((ratio,new_ratio))
    AUC_matrix = np.hstack((AUC_matrix,new_AUC_matrix))
    print("AUC_matrix =",AUC_matrix)
    X_train = new_X_train
    y_train = new_y_train
    
matrix_undersampling = AUC_matrix
ratio_undersampling = ratio


##plot the figure
plt.title('AUC Value Change by Undersampling')
plt.plot(ratio_undersampling[0,:], matrix_undersampling[0,:], 'r', label='Logistic Regression')
plt.plot(ratio_undersampling[0,:], matrix_undersampling[1,:], 'b', label='Random Forest')
plt.plot(ratio_undersampling[0,:], matrix_undersampling[2,:], 'g', label='Decision Tree')
plt.plot(ratio_undersampling[0,:], matrix_undersampling[3,:], 'y', label='KNN')
plt.plot(ratio_undersampling[0,:], matrix_undersampling[4,:], 'm', label='Naive Bayes')
plt.legend(loc='lower right')
plt.xlim([0.0,0.6])
plt.ylim([0.3,0.9])
plt.ylabel('AUC')
plt.xlabel('Label 1 Ratio')
plt.show()


##save the data to csv
table = np.vstack((matrix_undersampling, ratio_undersampling))
table = pd.DataFrame(table)    
table.to_csv('Undersampling_result.csv') 

