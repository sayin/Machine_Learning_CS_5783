# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:59:40 2019

@author: Harsha
"""


import numpy as np
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt


def distance(p1,p2):

    dx = p1[:,0] - p2.reshape((-1,p1.shape[1]))[:,0]
    dy = p1[:,1] - p2.reshape((-1,p1.shape[1]))[:,1]

    return np.sqrt(dx * dx + dy * dy)

def best_point(pt_, p1, p2):
    
    if p1 is None:
        return p2.reshape((-1,pt_.shape[1]))

    if p2 is None:
        return p1.reshape((-1,pt_.shape[1]))

    d1 = distance(pt_, p1)
    d2 = distance(pt_, p2)

    if d1 < d2:
        return p1.reshape((-1,pt_.shape[1]))
    else:
        return p2.reshape((-1,pt_.shape[1]))
    
    
def kdtree(data, depth=0):
##  
    n_samp = data.shape[0]
    if n_samp != 0:   
        axis = depth % data.shape[1]
        sort_data = data[data[:,axis].argsort()]
        
        return {'point': sort_data[n_samp//2,:],
             'left': kdtree(sort_data[:n_samp//2,:],depth+1),
            'right': kdtree(sort_data[n_samp//2 +1:,:], depth+1)}   


def search_nn(kd, pt, depth=0):
    
    if kd is not None:
        axis = depth % data.shape[1]
    
        next_branch     = None
        opposite_branch = None
    
        if pt[:,axis] < kd['point'][axis]:
            next_branch     = kd['left']
            opposite_branch = kd['right']
        else:
            next_branch     = kd['right']
            opposite_branch = kd['left']
    
        best = best_point(pt, search_nn(next_branch, pt, depth + 1), kd['point'])
        print(best)
        if distance(pt, best) > (pt[:,axis] - kd['point'][axis]):
            print('entered')
            best = best_point(pt, search_nn(opposite_branch, pt, depth + 1), best)
            print(best)
        return best


seed = 10
np.random.seed(seed)


samples = 10
features = 2
data = np.random.rand(samples, features) 
ytest= np.random.rand(1,features)


##Post

kdtree = kdtree(data)  
nn     = search_nn(kdtree, ytest)
plt.plot(data[:,0],data[:,1], 'b.', label= 'Training data')
plt.plot(nn[:,0],nn[:,1], 'g^', label = 'NN')
plt.plot(ytest[:,0],ytest[:,1], 'r*', label='Test data' )
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc = 'upper right')


# Validation
tree = cKDTree(data)
_,idx = tree.query(ytest,k=1)
nn_scipy = data[idx,:]

print('nn=',nn,'\n', 'nn_scipy=',nn_scipy)

