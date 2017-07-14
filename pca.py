# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:55:02 2017

@author: Yoran

principal components analysis
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y = np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])

mean_x = np.mean(x)
mean_y = np.mean(y)
scaled_x = x-mean_x
scaled_y = y-mean_y
data = np.matrix([[scaled_x[i],scaled_y[i]] for i in range(len(scaled_x))])

plt.plot(scaled_x,scaled_y,'o')

cov = np.cov(scaled_x,scaled_y)
eig_val, eig_vec = np.linalg.eig(cov)

xmin,xmax = scaled_x.min(),scaled_x.max()
ymin,ymax = scaled_y.min(),scaled_y.max()
dx = (xmax-xmin)*0.2
dy = (ymax-ymin)*0.2
plt.xlim(xmin-dx,xmax+dx)
plt.ylim(ymin-dy,ymax+dy)
plt.plot([eig_vec[:,0][0],0],[eig_vec[:,0][1],0],color='red')#eig_vec[:,0]取的是第一列
plt.plot([eig_vec[:,1][0],0],[eig_vec[:,1][1],0],color='blue')

eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort(reverse=True)
feature = eig_pairs[0][1]

new_data = np.transpose(np.dot(eig_vec,np.transpose(data)))
new_data_reduced = np.transpose(np.dot(feature,np.transpose(data)))
plt.plot(new_data[:,0],new_data[:,1],'^',color='red')
plt.plot(new_data_reduced[:,0],[1.2]*10,'*',color='green')

