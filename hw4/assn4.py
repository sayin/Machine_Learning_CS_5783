#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:17:01 2019

@author: harsha
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.random import seed
seed(0)

def kernel_EX(a, b, param):
    absdist = np.abs(a - b.reshape(-1,))
    return np.exp(-1.0 * (1.0/param) * absdist)

def kernel_SE(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2.0*np.dot(a, b.T)
    return np.exp(-0.5 * (1.0/param) * sqdist)


def gp_kernel_EX(xtrain,ytrain,xtest,param):
    

     K = kernel_EX(xtrain, xtrain, param)
     C = K + (0.09569**2)*np.identity(xtrain.shape[0]) 
     K_s = kernel_EX(xtrain, xtest, param)
     mu = np.dot(K_s.T, np.dot(np.linalg.inv(C),ytrain))
    
     return mu

def gp_kernel_SE(xtrain,ytrain,xtest,param):
       
    K = kernel_SE(xtrain, xtrain, param)
    L = np.linalg.cholesky(K + (0.09569**2)*np.eye(len(xtrain)))
    K_s = kernel_SE(xtrain, xtest, param)
    Lk = np.linalg.solve(L, K_s)
    mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((xtest.shape[0],))
        
    return mu
    
#%%
def problem1():
    data = np.loadtxt('crash.txt')
    m,n = data.shape
    
    data_min = np.min(data,axis=0)
    data = data - data_min
    data_max = np.max(data,axis=0)
    data = data/data_max
    
    kfold = 5
    cvSE = np.zeros((100,6))
    cvEX = np.zeros((100,6))
    params = np.linspace(0.01,0.5,100)
    foldsize = int(m/kfold)
    
    rIDX = np.arange(m)
    np.random.shuffle(rIDX)
    
    for j in range(params.shape[0]):
        param = params[j]
        cvSE[j,0] = param
        
        for i in range(5):
            ridx_train = np.vstack((rIDX[:i*foldsize].reshape(-1,1),
                                       rIDX[(i+1)*foldsize:].reshape(-1,1)))
            
            ridx_test = rIDX[i*foldsize:(i+1)*foldsize]
            
            x_train = data[ridx_train.reshape(-1,),0].reshape(-1,1)
            y_train = data[ridx_train.reshape(-1,),1].reshape(-1,1)
            
            x_test = data[ridx_test.reshape(-1,),0].reshape(-1,1)
            y_test = data[ridx_test.reshape(-1,),1].reshape(-1,1)
            
            muEX = gp_kernel_EX(x_train,y_train,x_test,param)
            mseEX = np.mean((np.square(muEX - y_test.reshape(-1,))))
            cvEX[j,i+1] = mseEX 

            muSE = gp_kernel_SE(x_train,y_train,x_test,param)
            mseSE = np.mean((np.square(muSE - y_test.reshape(-1,))))
            cvSE[j,i+1] = mseSE        

    mseEX = np.mean(cvEX[:,1:],axis=1)        
    optEX = params[np.argmin(mseEX)]
    
    mseSE = np.mean(cvSE[:,1:],axis=1)        
    optSE = params[np.argmin(mseSE)]
    
    
    x_train, y_train = data[:,0].reshape(-1,1), data[:,1].reshape(-1,1)
    x_test = np.linspace(0,1,100)
    x_test = x_test.reshape(-1,1)
    
    muSE= gp_kernel_SE(x_train,y_train,x_test,optSE)
    muEX= gp_kernel_EX(x_train,y_train,x_test,optEX)
    
    print('Squared exponential')
    print('Optimal sigma = ', format(optSE, '.2f'))
    plt.figure(figsize = (10,4))
    plt.plot(x_test,muSE,color='b',marker='o',label='Gaussian process')
    plt.plot(data[:,0],data[:,1],color='g',marker='o',label='Real data')
    plt.legend()
    plt.show()
    
    print('Exponential')
    print('Optimal sigma = ', format(optEX, '.2f'))
    plt.figure(figsize = (10,4))
    plt.plot(x_test,muEX,color='b',marker='o',label='Gaussian process')
    plt.plot(data[:,0],data[:,1],color='g',marker='o',label='Real data')
    plt.legend()
    plt.show()

#problem1()

#%%
def imagedata(n):
    IMG_file = open('t10k-images-idx3-ubyte','rb')
    IMG = IMG_file.read()
    IMG = bytearray(IMG)
    IMG = IMG[16:]
    
    LAB_file = open('t10k-labels-idx1-ubyte','rb')
    LAB = LAB_file.read()
    LAB = bytearray(LAB)
    LAB = LAB[8:]
    
    IMG = np.array(IMG)
    IMG = IMG.reshape(n,-1)
    IMG = IMG.astype(np.float64)
    
    LAB = np.array(LAB)
    LAB = LAB.reshape(n,-1)
    
    return IMG, LAB 

def problem2(): 
    n=10000
    dataIMG, dataLAB = imagedata(n) 
    
    K = 10
    rIDX = np.random.randint(0,n,K)
    centroid = dataIMG[rIDX] 

    centroid = np.zeros((K,dataIMG.shape[1]))
    centroid[0,:] = dataIMG[np.random.randint(0,n,1)]
    for k in range(1,K):
        b = dataIMG - centroid[k-1,:]
        distk = LA.norm(b,axis=1)**2
        centroid[k,:] = dataIMG[np.argmax(distk)]

    centroid = np.zeros((K,dataIMG.shape[1]))
    for k in range(K):
        centroid[k,:] = dataIMG[dataLAB.reshape(-1,)==k][0]

    K = 3
    centroid = np.zeros((K,dataIMG.shape[1]))
    
    centroid[0,:] = dataIMG[np.random.randint(0,n,1)]
    for k in range(1,K):
        b = dataIMG - centroid[k-1,:]
        distk = LA.norm(b,axis=1)**2
        centroid[k,:] = dataIMG[np.argmax(distk)]
        
        ctemp =  np.zeros((K,dataIMG.shape[1]))
        dmat = np.zeros((dataIMG.shape[0],K))
        res = 1
        n = 0
        
        while res  > 0.001:
            for k in range(K):
                b = dataIMG - centroid[k,:]
                dmat[:,k] = LA.norm(b,axis=1)**2
            
            ind = np.argmin(dmat,axis=1)
        
            for k in range(K):
                temp = dataIMG[ind == k]
                if temp.size != 0:
                    ctemp[k,:] = np.average(temp,axis=0)
            
            res= LA.norm(centroid - ctemp)
            centroid[:,:] = ctemp[:,:]
            print('.',end =" ")
            n = n+1
        
        T = 0
        for k in range(K):
            temp = dataIMG[ind == k]
            b = temp - centroid[k,:]
            T = T + LA.norm(b)**2
        
        print(n, " Iterations ", T )
        
        p=3
        if p == 3:
            unique, counts = np.unique(ind, return_counts=True)
    
            fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(6,3))
            ax = axs.flat
            
            for k in range(K):
                ax[k].imshow(centroid[k].reshape(28,28), cmap='gray')
                ax[k].set_xticks([])
                ax[k].set_yticks([])
            
            fig.tight_layout()
            print('See plot.')
            plt.show()
            
            fig, axs = plt.subplots(nrows=3,ncols=2,figsize=(6,9))
            ax = axs.flat
            p = 0
            
            for i in range(3):
                for j in range(2):
                    ax[p].imshow(dataIMG[ind==unique[i]][j].reshape(28,28),cmap='gray')
                    ax[p].set_xticks([])
                    ax[p].set_yticks([])
                    ax[p].set_title('Cluster '+str(i))
                    p = p+1
            fig.tight_layout()
            print('See plot.')
            plt.show()
            
        else:        
            fig, axs = plt.subplots(nrows=2,ncols=5,figsize=(10,4))
            ax = axs.flat
            
            for k in range(K):
                ax[k].imshow(centroid[k].reshape(28,28),cmap='gray')
                ax[k].set_xticks([])
                ax[k].set_yticks([])
            
            fig.tight_layout()
            print('See plot.')
            plt.show()

#problem2()
#%%
def forward(V, a, b, intit_dist):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = intit_dist * b[:, int(V[0]-1)]
    
    alpha[0, :] = alpha[0,:]/np.sum(alpha[0,:])
    
    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, int(V[t]-1)]
        alpha[t,:]  = alpha[t,:]/np.sum(alpha[t,:])
        
    return alpha
 

def backward(Pr, a, b, alpha):
    beta = np.zeros((Pr.shape[0], a.shape[0]))
 

    beta[Pr.shape[0] - 1] = np.ones((a.shape[0]))
    beta[Pr.shape[0] - 1,:]  = beta[-1,:]/np.sum(beta[-1,:])
 

    for t in range(Pr.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, int(Pr[t + 1]-1)]).dot(a[j, :])
        beta[t,:]  = beta[t,:]/np.sum(beta[t,:])
        
    return beta 

def problem3():
    seed(0)
    latVAR = np.array([0,1])
    xOBS = np.array([1,2,3,4,5,6]) 
    
    probZ = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])
    FL = np.array([[0.95, 0.05],[0.1,0.9]])                
    
    N = 1000
    x_true = np.zeros(N)
    ptrue = np.zeros(N)
    
    ptrue[0] = 0
    px = probZ[int(ptrue[0])]
    x_true[0] = np.random.choice(xOBS, 1, p=px)
    
    for i in range(1,N):
       
       pz = FL[int(ptrue[i-1])]
       ptrue[i] = np.random.choice(latVAR, 1, p=pz)
       
       px = probZ[int(ptrue[i])]
       x_true[i] = np.random.choice(xOBS, 1, p=px)
    
    transP= np.array(((0.95, 0.05), (0.1, 0.9)))
    emisP = np.array(((1/6, 1/6, 1/6, 1/6, 1/6, 1/6), (0.1, 0.1, 0.1, 0.1, 0.1, 0.5)))
    intit_dist = np.array((0.5,0.5))   
     
    alpha = forward(x_true, transP, emisP, intit_dist)
    beta = backward(x_true, transP, emisP, alpha)
    posterior = alpha*beta
    posterior = posterior[:,:]/np.sum(posterior,axis=1,keepdims=True)
    
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(13,4))
    ax.plot(ptrue,color='gray',label='Actual Loaded Dice')
    ax.plot(alpha[:,1],'b',label='Probability of Loaded Dice')
    ax.legend(loc=2)
    ax.set_xlabel('Time steps')
    ax.set_ylabel('P(loaded)')
    ax.set_title('Forward Step')
    plt.show()
    
    
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(13,4))
    ax.plot(ptrue,color='gray',label='Actual Loaded Dice')
    ax.plot(beta[:,1],'b',label='Probability of Loaded Dice')
    ax.legend(loc=2)
    ax.set_xlabel('Time steps')
    ax.set_ylabel('P(loaded)')
    ax.set_title('Posterior')
    plt.show()
    
#problem3()    