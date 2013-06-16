from sklearn.neighbors import NearestNeighbors
from random import choice

'''
Created on 14-jun.-2013

@author: Olivier.Janssens
'''

import numpy as np
import random

# @param value: The classlabel
# @param qlist: The list in which to search
# @return: the indices of the values that are equal to the classlabel
def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx+1)
            indices.append(idx)
        except ValueError:
            break
    return indices

# @param: X The datapoints e.g.: [f1, f2, ... ,fn]
# @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
# @return ms: The amount of samples in the minority group
# @return ms: The amount of samples in the majority group
def getClassCount(X,y):
    indicesZero = all_indices(0,y); 
    indicesOne = all_indices(1,y);
    
    if(len(indicesZero)>len(indicesOne)):
        ms = len(indicesOne)
        ml = len(indicesZero)
    else:
        ms = len(indicesZero)
        ml = len(indicesOne)
    return ms,ml

# @param: X The datapoints e.g.: [f1, f2, ... ,fn]
# @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
# @param ms: The amount of samples in the minority group
# @param ms: The amount of samples in the majority group
# @return: The ratio between the minority and majority group
def getd(X,y,ms,ml):

    return float(ms)/float(ml)

# @param: X The datapoints e.g.: [f1, f2, ... ,fn]
# @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
# @param ms: The amount of samples in the minority group
# @param ms: The amount of samples in the majority group
# @return: the G value, which indicates how many samples should be generated in total, this can be tuned with beta
def getG(X,y,ms,ml,beta):
    return (ml-ms)*beta


# @param: X The datapoints e.g.: [f1, f2, ... ,fn]
# @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
# @param: minorityclass: The minority class
# @param: K: The amount of neighbours for Knn
# @return: rlist: List of r values
def getRis(X,y,minorityclass,K):
    indicesMinority = all_indices(minorityclass,y)
    ymin = np.array(y)[indicesMinority]
    Xmin = np.array(X)[indicesMinority]
    neigh = NearestNeighbors(n_neighbors=30,algorithm = 'ball_tree')
    neigh.fit(X)
    
    rlist = [0]*len(ymin)
    normalizedrlist = [0]*len(ymin)
    
    for i in xrange(len(ymin)):
         indices = neigh.kneighbors(Xmin[i],K,False)
         rlist[i] = float(len(all_indices(1,np.array(y)[indices].tolist()[0])))/K

        
    normConst = sum(rlist)

    for j in xrange(len(rlist)):
        normalizedrlist[j] = (rlist[j]/normConst)

    return normalizedrlist
        

# @param: rlist: List of r values
# @param: X The datapoints e.g.: [f1, f2, ... ,fn]
# @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
# @return: the G value, which indicates how many samples should be generated in total, this can be tuned with beta
# @param: minorityclass: The minority class
# @param: K: The amount of neighbours for Knn
# @return: The synthetic data samples
def generateSamples(rlist,X,y,G,minorityclasslabel,K):
    syntheticdata = []
    
    indicesMinority = all_indices(minorityclasslabel,y)
    ymin = np.array(y)[indicesMinority]
    Xmin = np.array(X)[indicesMinority]
    
    
    neigh = NearestNeighbors(n_neighbors=30,algorithm = 'ball_tree')
    neigh.fit(Xmin)
    
    for k in xrange(len(ymin)):
        g = int(np.round(rlist[k]*G))
        
        for l in xrange(g):
            ind = random.choice(neigh.kneighbors(Xmin[k],K,False)[0])
            s = Xmin[k] + (Xmin[ind]-Xmin[k]) * random.random()
            syntheticdata.append(s)
    
    newData = np.concatenate((syntheticdata,Xmin),axis=0)
    newy = [0]*len(newData)
    return newData,newy


def joinwithmajorityClass(X,y,newData,newy,majorityclasslabel):
    indicesMajority = all_indices(majorityclasslabel,y)
    ymaj = np.array(y)[indicesMajority]
    Xmaj = np.array(X)[indicesMajority]
    
    
    return np.concatenate((Xmaj,newData),axis=0),np.concatenate((ymaj,newy),axis=0)
    
    
        
        
    