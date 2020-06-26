# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 10:01:00 2019

@author: Harsha
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
np.random.seed(100)

def problem1():
    data = np.loadtxt( 'crash.txt')
    train = data[::2,:]
    validation = data[1::2,:]
    
    t_train = train[:,1].reshape((-1,1))
    x_train = train[:,0].reshape((-1,1))
    
    t_test = validation[:,1].reshape((-1,1))
    x_test = validation[:,0].reshape((-1,1))
    
    L = np.linspace(0,22,23)
    N = x_train.shape[0]
    M = x_test.shape[0]
    
    rmsT = []   
    rmsV = [] 
    t_pred = np.empty((t_test.shape[0],0))
      
    for i in range(len(L)):    
        
        phi_train = np.zeros((N,i))
        phi_test  = np.zeros((M,i))
        phi_train =  x_train**[L[:i+1]]
        phi_test  =  x_test**[L[:i+1]]
        
        a = np.dot(phi_train.T,phi_train)
        b = np.dot(phi_train.T,t_train)   
        w = np.linalg.solve(a,b)
        
        t_pred_tr =np.dot(phi_train,w)
        rms1 = np.sqrt(((t_pred_tr-t_train)**2).mean(axis=0))
        
        t_pred_ts =np.dot(phi_test,w)
        t_pred = np.hstack((t_pred, t_pred_ts))
        rms2 = np.sqrt(((t_pred_ts - t_test)**2).mean(axis=0))
        
        rmsT.append(rms1)
        rmsV.append(rms2) 
        
    print('See plot.')    
     
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(15,6))
    
    ax[0].plot(L[1:],rmsT[1:],'b',label='Training')
    ax[0].plot(L[1:],rmsV[1:],'r',label='Validation')
    ax[0].set_xticks(np.arange(1,23,1))    
    
    ax[0].legend(loc=0)
    
    ax[1].scatter(x_train,t_train,color='k',label='Training data')
    ax[1].scatter(x_test,t_test,color='g',label='Validation Data')
    ax[1].plot(x_test,t_pred[:,np.argmin(rmsV)],color='r',label='Validation Best Fit')
    ax[1].legend(loc=4)
    
    plt.show()

#problem1()  

#%%

def problem2():
    data = np.loadtxt( 'crash.txt')
    train = data[::2,:]
    validation = data[1::2,:]
    
    t_train = train[:,1].reshape((-1,1))
    x_train = train[:,0].reshape((-1,1))
    
    t_test = validation[:,1].reshape((-1,1))
    x_test = validation[:,0].reshape((-1,1))
    
    L = np.linspace(5,25,5)
    N = x_train.shape[0]
    M = x_test.shape[0]
        
    rmsT = []   
    rmsV = []  
    t_pred = np.empty((t_test.shape[0],0))
      
    for i in L:  
        phi_train = np.zeros((N,int(i)))
        phi_test  = np.zeros((M,int(i)))
        mean = np.linspace(min(x_train),max(x_train),int(i))
        std = mean[1] - mean[0]
        
        phi_train1 =  np.exp(-(x_train-mean)**2/(2*std**2))
        phi_test1  =  np.exp(-(x_test-mean)**2/(2*std**2))
        
        phi_train = np.column_stack(((np.ones(N,)),phi_train1))
        phi_test = np.column_stack(((np.ones(M,)),phi_test1))
        
        a = np.dot(phi_train.T,phi_train)
        b = np.dot(phi_train.T,t_train)   
        w = np.linalg.solve(a,b)
        
        t_pred_tr =np.dot(phi_train,w)
        rms1 = np.sqrt(((t_pred_tr-t_train)**2).mean(axis=0))
        
        t_pred_ts =np.dot(phi_test,w)
        t_pred = np.hstack((t_pred, t_pred_ts))
        rms2 = np.sqrt(((t_pred_ts - t_test)**2).mean(axis=0))
        
        rmsT.append(rms1)
        rmsV.append(rms2)
        
    print('See plot.')    
        
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(15,8))
    
    ax[0].plot(L,rmsT,'b',label='Training')
    ax[0].plot(L,rmsV,'r',label='Validation')
    ax[0].set_xticks(np.arange(5,26,5))    
    
    ax[0].legend(loc=0)
    
    ax[1].scatter(x_train,t_train,color='k',label='Training data')
    ax[1].scatter(x_test,t_test,color='g',label='Validation Data')
    ax[1].plot(x_test,t_pred[:,np.argmin(rmsV)],color='r',label='Validation Best Fit')
    ax[1].legend(loc=4)
    
    plt.show()

#problem2()
#
#%%
def problem3():
    data = np.loadtxt( 'crash.txt')
    train = data[::2,:]
    validation = data[1::2,:]
    
    t_train = train[:,1].reshape((-1,1))
    x_train = train[:,0].reshape((-1,1))
    
    t_test = validation[:,1].reshape((-1,1))
    x_test = validation[:,0].reshape((-1,1))
    
    L = 50
    beta = 0.0025
    alpha = np.logspace(-8,0,100)
    N = x_train.shape[0]
    M = x_test.shape[0]
 
    rmsT = []   
    rmsV = []  
    t_pred = np.empty((t_test.shape[0],0))
      
    for i in range(alpha.shape[0]):  
        phi_train = np.zeros((N,L))
        phi_test  = np.zeros((M,L))
        mean = np.linspace(0,60,L)
        std = mean[1] - mean[0]
        
        phi_train =  np.exp(-(x_train-mean)**2/(2*std**2))
        phi_test  =  np.exp(-(x_test-mean)**2/(2*std**2))
        
        a = np.dot(phi_train.T,phi_train) + alpha[i]/beta*(np.eye(L)) 
        b = np.dot(phi_train.T,t_train)   
        w = np.linalg.solve(a,b)
        
        t_pred_tr =np.dot(phi_train,w)
        rms1 = np.sqrt(((t_pred_tr-t_train)**2).mean(axis=0))
        
        t_pred_ts =np.dot(phi_test,w)
        t_pred = np.hstack((t_pred, t_pred_ts))
        rms2 = np.sqrt(((t_pred_ts - t_test)**2).mean(axis=0))
        
        rmsT.append(rms1)
        rmsV.append(rms2)

    print('Best alpha:',alpha[np.argmin(rmsV)])
    print('See plot.')    
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(15,8))
    ax.scatter(x_train,t_train,color='k',label='Training data')
    ax.scatter(x_test,t_test,color='g',label='Validation Data')
    ax.plot(x_test,t_pred[:,np.argmin(rmsV)],color='r',label='Validation Best Fit')
    ax.legend(loc=4)
    plt.show()  

#problem3()

#%%
def problem4():
    
    def flower_to_float(s):
        d = {b'Iris-setosa':0., b'Iris-versicolor':1., b'Iris-virginica':2.}
        return d[s]
    
    irises = np.loadtxt( 'iris.data' , delimiter=',' , converters = {4:flower_to_float} )
    
    K = 3
    M = 5
    N = 150
    
    label = irises[:,-1]
    data = np.column_stack((np.ones(N), irises[:,:-1]))
    
    oh = np.zeros((N,3))
    oh[np.arange(N),label.astype(int)] = 1
    
    xtrain = data[1::2]
    xtest = data[::2]
    
    ytrain = oh[1::2]
    ytest  = oh[::2]
    
    w_init = np.ones((M*K,1))
    
    alpha = 0.003
    
    def f(w, data, t):  
        
        pw = (alpha/2)*(np.dot(w.T,w))
        a = 0
        b = 0   
        k_ = 5
        for i in range(3):
            atemp = np.exp(np.dot(w[k_-5:k_].T,xtrain.T)) 
            btemp = ytrain[:,i]*np.dot(w[k_-5:k_].T,xtrain.T)
            a = atemp + a
            b = btemp + b
            k_ = k_+5
    
        a2 = np.log(a)
        c = b-a2    
        j = pw - np.sum(c)
            
        return j.reshape((-1,1))
    
    w_hat = minimize(f, w_init, args=(xtrain, ytrain)).x
    
    yp1 = np.dot(w_hat[:5].T,xtest.T)
    yp2 = np.dot(w_hat[5:10].T,xtest.T)
    yp3 = np.dot(w_hat[10:].T,xtest.T)
    
    ypred1 = np.exp(yp1)/(np.sum(np.exp(yp1) + np.exp(yp2) + np.exp(yp3))) 
    ypred2 = np.exp(yp2)/(np.sum(np.exp(yp1) + np.exp(yp2) + np.exp(yp3))) 
    ypred3 = np.exp(yp3)/(np.sum(np.exp(yp1) + np.exp(yp2) + np.exp(yp3))) 
    
    ypred = np.stack((ypred1,ypred2, ypred3),axis=1)
    
    ycls1 = ypred.max(axis=1).reshape(-1, 1)
    ycls = np.where(ypred == ycls1, 1, 0) 
    
    count1 = np.sum(ycls[:25,0])
    count2 = np.sum(ycls[25:50,1])
    count3 = np.sum(ycls[50:75,2])
    
    cls_acc = (count1+count2+count3)/ytest.shape[0]
    print('Logistic regression accuracy on test set:',cls_acc)    
    
    
#problem4()    