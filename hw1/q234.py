#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 09:51:19 2019

@author: sayin
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score

seed = 10
np.random.seed(seed)    
    
def comp_mean(size):
    mean = np.random.rand(dim)
    return mean


def comp_cov(size):
    a =  np.random.rand(size,size)
    cov = np.dot(a,a.T)
    return cov


def multi_gauss_dist(dim,samples):
    cov = comp_cov(dim)
    mean = comp_mean(dim)
    data = np.random.multivariate_normal(mean, cov, samples)
    return data

def random_data(x_, y_):
    data = np.column_stack((x_,y_))
    np.random.shuffle(data)
    
    return data[:,:-1],data[:,-1] 
    
def split_train_test(x_,y_,split):
    msk = np.random.rand(len(y_)) < split
    xtrain_ = x_[msk]
    ytrain_ = y_[msk]
    
    xtest_ =  x_[~msk]
    ytest_ = y_[~msk] 
    
    return xtrain_, ytrain_, xtest_, ytest_

def clasf_acc(ytst, yc):
    
    bol = np.where(ytest.reshape((-1,1)) == yc.reshape((-1,1)), 1, 0 )
    acc = np.count_nonzero(bol)/len(yc)
    print('Classification accuracy:%.5f'%acc)    
    #    accuracy_score(ytest, kycls)
    return acc


def cls_utility(xtrain, ytrain, xtest, ytest, ycls):

    bol0_train = ytrain==0
    cls0_train = xtrain[bol0_train]
    cls1_train = xtrain[~bol0_train]
    
    bol0_test = ytest == 0
    
    cls0_test  = ytest[bol0_test]
    cls0_cls   = ycls[bol0_test]
    cls0_xtest = xtest[bol0_test]
     
    cls1_test   = ytest[~bol0_test]
    cls1_cls    = ycls[~bol0_test]
    cls1_xtest  = xtest[~bol0_test]
    
    
    bol0_true   = cls0_test == cls0_cls.reshape((-1,))
    
    cls0_true   = cls0_xtest[bol0_true]
    cls0_false  = cls0_xtest[~bol0_true]
    
    bol1_true   = cls1_test == cls1_cls.reshape((-1,))
    
    cls1_true   = cls1_xtest[bol1_true]
    cls1_false  = cls1_xtest[~bol1_true]
    
    return cls0_train, cls1_train, cls0_true, cls0_false, cls1_true, cls1_false


def post(x, y, xt, xf, yt, yf, q, c):    
    
    if q==2:
        
        plt.plot(x[:,0],x[:,1], '.', label='Class 0')
        plt.plot(y[:,0], y[:,1], '+', label='Class 1')
        
    else:
        plt.plot(x[:,0], x[:,1], '.', label='Class 0 train')
        plt.plot(y[:,0], y[:,1], '+', label='Class 1 train')
#        
        plt.plot(xt[:,0], xt[:,1], 'g.', label='Class 0 True')
        plt.plot(xf[:,0], xf[:,1], 'r.', label='Class 0 False')
    
        plt.plot(yt[:,0], yt[:,1], 'g+', label='Class 1 True')
        plt.plot(yf[:,0], yf[:,1], 'r+', label='Class 1 False')
        
    if c == 'l':        
        plt.title('Linear Classifier')
    elif c == 'k' :
        plt.title('kD Classifier')
        
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    
    
    
## Construct X (input) and y (target)    
dim = 2
n_samp = 5000

x1 = multi_gauss_dist(dim,n_samp) 
x2 = multi_gauss_dist(dim,n_samp) 

x = np.vstack((x1,x2))
y = np.vstack((np.zeros((n_samp,1)),np.ones((n_samp,1)))) 
 
xdata,ydata = random_data(x, y)

xtrain, ytrain, xtest, ytest = split_train_test(xdata,ydata,split=0.8)

#%%
## Linear Classifier

def lin_class(xtr, ytr):
    t0 = np.linalg.inv(np.dot(xtr.T,xtr))
    t1 = np.dot(t0,xtr.T)
    beta = np.dot(t1,ytr)
    
    return beta

def lin_class_pred(beta,xts, tol):
    
    ypred_ = np.dot(xts, beta.reshape((-1,1)))    
    ycls_ = np.where(ypred_ < tol, 0,1 )
    
    return ypred_, ycls_    

def kd_class(xtr, xts):
    tree = cKDTree(xtr)
    _,idx = tree.query(xts,k=1)
    return idx  

def kd_class_pred(ytr, idx):
    ycls_ = ytrain[idx]
    return ycls_

#%%   Linear Class
beta = lin_class(xtrain, ytrain)
ypred, lycls = lin_class_pred(beta, xtest, tol=0.5)

accuracy = clasf_acc(ytest, lycls)
    
lcls0_train, lcls1_train, lcls0_true, lcls0_false, lcls1_true, lcls1_false = cls_utility(xtrain, ytrain, xtest, ytest, lycls)

plt.figure()
q=3
post(lcls0_train, lcls1_train, lcls0_true, lcls0_false, lcls1_true, lcls1_false,q,c='l' )    


#%% kD tree
idx = kd_class(xtrain, xtest)
kycls = kd_class_pred(ytrain, idx)

accuracy = clasf_acc(ytest, kycls)

kcls0_train, kcls1_train, kcls0_true, kcls0_false, kcls1_true, kcls1_false = cls_utility(xtrain, ytrain, xtest, ytest, kycls)

plt.figure()
post(kcls0_train, kcls1_train, kcls0_true, kcls0_false, kcls1_true, kcls1_false, q, c='k')    









