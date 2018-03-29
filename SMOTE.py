#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 15:39:09 2017

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
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_clean_data():
    data=pd.read_csv('cs-training.csv', sep=',',header = 0)
    ##remove missing value row
    data_withoutna = data.dropna(axis = 0, how = "any")
    data_withoutna = np.array(data_withoutna.iloc[:,1:])
    
    ##remove outliers of each feature
    new_data1 = data_withoutna[(np.abs(stats.zscore(data_withoutna[:,1])) < 3)]
    new_data2 = new_data1[(new_data1[:,2] < 90) & (new_data1[:,2] > 20)]
    new_data3 = new_data2[new_data2[:,3] < 50]
    new_data4 = new_data3[(new_data3[:,4] < 1) & (new_data3[:,4] > 0)]
    new_data5 = new_data4[(np.abs(stats.zscore(new_data4[:,5])) < 3)]
    data_set = new_data5[:,1:]
    data_label = new_data5[:,0]
    X_train, X_test, y_train, y_test = train_test_split(data_set, data_label, test_size=0.25, random_state = 0)
    return X_train, X_test, y_train, y_test

##logistic regression
def logreg(X_train, y_train, X_test, y_test,draw):
    logreg = linear_model.LogisticRegression()  
    logreg.fit(X_train, y_train)  
    predictions=logreg.predict_proba(X_test)
    false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:
        , 1])
    roc_auc=auc(false_positive_rate,recall)
    if draw == 1:
        plt.title('ROC Curve')
        plt.plot(false_positive_rate, recall, 'r', label='LG AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.ylabel('Recall')
        plt.xlabel('Fall-out')
        plt.show()
    return roc_auc

##random forest
def rf(X_train, y_train, X_test, y_test,draw):
    rf = RandomForestClassifier()  
    rf.fit(X_train,y_train)
    predictions=rf.predict_proba(X_test)
    false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:
        , 1])
    roc_auc=auc(false_positive_rate,recall)
    if draw == 1:
        plt.title('ROC Curve')
        plt.plot(false_positive_rate, recall, 'b', label='RF AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.ylabel('Recall')
        plt.xlabel('Fall-out')
        plt.show()
    return roc_auc


##decision tree
def decitree(X_train, y_train, X_test, y_test,draw):
    clf = tree.DecisionTreeClassifier(criterion='entropy')  
    clf.fit(X_train, y_train) 
    predictions=clf.predict_proba(X_test)
    false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:
        , 1])
    roc_auc=auc(false_positive_rate,recall)
    if draw == 1:
        plt.title('ROC Curve')
        plt.plot(false_positive_rate, recall, 'g', label='DT AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.ylabel('Recall')
        plt.xlabel('Fall-out')
        plt.show()
    return roc_auc


##knn
def knn(X_train, y_train, X_test, y_test,draw):
    knn = neighbors.KNeighborsClassifier()  
    knn.fit(X_train, y_train) 
    predictions=knn.predict_proba(X_test)
    false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:
        , 1])
    roc_auc=auc(false_positive_rate,recall)
    if draw == 1:
        plt.title('ROC Curve')
        plt.plot(false_positive_rate, recall, 'y', label='KNN AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.ylabel('Recall')
        plt.xlabel('Fall-out')
        plt.show()
    return roc_auc 


##naive bayes
def nb(X_train, y_train, X_test, y_test, draw):
    nb = GaussianNB()    
    nb.fit(X_train, y_train)   
    predictions=nb.predict_proba(X_test)
    false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:
        , 1])
    roc_auc=auc(false_positive_rate,recall)
    if draw == 1:
        plt.title('ROC Curve')
        plt.plot(false_positive_rate, recall, 'm', label='NB AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.ylabel('Recall')
        plt.xlabel('Fall-out')
        plt.show()
    return roc_auc 


##euclidean distance
def dist(a, b):
    a_norm = (a**2).sum()
    b_norm = (b**2).sum()
    dist = a_norm + b_norm - 2*a.dot(b.transpose())
    return dist


##smote algorithm
def smote(label,N,K,X_train,y_train):
    aimed_X_train = X_train[np.argwhere(y_train == label).squeeze()]
    index = np.random.choice(len(aimed_X_train),round(len(aimed_X_train)*N),replace = False)
    aim_data = aimed_X_train[index]
    sc = StandardScaler().fit(aim_data)
    sc_aim_data = sc.transform(aim_data)
    dis_matrix = np.zeros((len(sc_aim_data),len(sc_aim_data)))
    for i in range(len(dis_matrix)):
        for j in range(len(dis_matrix)):
            dis_matrix[i,j] = dist(sc_aim_data[i],sc_aim_data[j])
    dis_matrix = dis_matrix + 10000 * np.eye(len(dis_matrix))    #set the diagonal value to 10000 to avoid to be selected as K neighbor
    index_matrix = np.zeros((len(dis_matrix),K))
    for l in range(len(dis_matrix)):
        index_matrix[l] = np.argsort(dis_matrix[l])[0:K]
    chosen_index = np.zeros((len(dis_matrix),1))
    for l in range(len(dis_matrix)):
        chosen_index[l] = np.random.choice(index_matrix[l],1)
    new_data6 = np.zeros((len(dis_matrix),10))
    for l in range(len(dis_matrix)):
        new_data6[l,:] = aim_data[l,:] + np.random.rand() * (aim_data[chosen_index[l,0].astype(int),:] - aim_data[l,:])
    new_X_train = np.vstack((X_train, new_data6))
    new_y_train = np.vstack((y_train.reshape(len(y_train),1),np.ones((len(new_data6),1))))
    new_dataset = np.hstack((new_X_train,new_y_train))
    np.random.shuffle(new_dataset)
    return new_dataset

X_train, X_test, y_train, y_test = load_clean_data()
##calculate the original AUC
original_X_train = X_train
original_y_train = y_train
AUC_matrix = np.zeros((5,1))
draw = 1
AUC_matrix[0,0] = logreg(original_X_train, original_y_train, X_test, y_test, draw)
AUC_matrix[1,0] = rf(original_X_train, original_y_train, X_test, y_test, draw)
AUC_matrix[2,0] = decitree(original_X_train, original_y_train, X_test, y_test, draw)
AUC_matrix[3,0] = knn(original_X_train, original_y_train, X_test, y_test, draw)
AUC_matrix[4,0] = nb(original_X_train, original_y_train, X_test, y_test, draw)
draw = 0
ratio = np.zeros((1,1))
ratio[0,0] = sum(original_y_train)/len(original_y_train)

##calculate the change of AUC under SMOTE with different label 1 ratio
indicator = ratio[0,0]
while indicator < 0.7:
    new_dataset = smote(1,0.1,5,X_train,y_train)
    new_X_train = new_dataset[:,:10]
    new_y_train = new_dataset[:,10]
    new_AUC_matrix = np.zeros((5,1))
    new_ratio = np.zeros((1,1))
    new_AUC_matrix[0,0] = logreg(new_X_train, new_y_train, X_test, y_test, draw)
    new_AUC_matrix[1,0] = rf(new_X_train, new_y_train, X_test, y_test, draw)
    new_AUC_matrix[2,0] = decitree(new_X_train, new_y_train, X_test, y_test, draw)
    new_AUC_matrix[3,0] = knn(new_X_train, new_y_train, X_test, y_test, draw)
    new_AUC_matrix[4,0] = nb(new_X_train, new_y_train, X_test, y_test, draw)
    new_ratio[0,0] = sum(new_y_train)/len(new_y_train)
    indicator = new_ratio[0,0]
    print(indicator)
    ratio = np.hstack((ratio,new_ratio))
    AUC_matrix = np.hstack((AUC_matrix,new_AUC_matrix))
    X_train = new_X_train
    y_train = new_y_train
    
       
matrix_over = AUC_matrix
ratio_over = ratio

##plot the figure of AUC change
plt.title('AUC Value Change by SMOTE')
plt.plot(ratio_over[0,:], matrix_over[0,:], 'r', label='Logistic Regression')
plt.plot(ratio_over[0,:], matrix_over[1,:], 'b', label='Random Forest')
plt.plot(ratio_over[0,:], matrix_over[2,:], 'g', label='Decision Tree')
plt.plot(ratio_over[0,:], matrix_over[3,:], 'y', label='KNN')
plt.plot(ratio_over[0,:], matrix_over[4,:], 'm', label='Naive Bayes')
plt.legend(loc='lower right')
plt.xlim([0.0,0.6])
plt.ylim([0.3,0.9])
plt.ylabel('AUC')
plt.xlabel('Label 1 Ratio')
plt.show()

##save the result to csv
table = np.vstack((matrix_over, ratio_over))
table = pd.DataFrame(table)    
table.to_csv('SMOTE_result.csv') 



