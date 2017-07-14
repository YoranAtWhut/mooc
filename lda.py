# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:58:37 2017

@author: Yoran
"""

import numpy as np
import csv
from matplotlib import pyplot as plt
import math

def read_iris():
    from sklearn.datasets import load_iris
    from sklearn import preprocessing
    data_set = load_iris()
    data_x = data_set.data
    label = data_set.target + 1
    #对数据进行标准化
    preprocessing.scale(data_x,axis=0,with_mean=True,with_std=True,copy=False)
    return data_x, label

def class_mean(data,label,clusters):
    mean_vectors = []
    for cl in range(1,clusters+1):
        mean_vectors.append(np.mean(data[label==cl,],axis=0))
    #print(mean_vectors)
    return mean_vectors

def within_class_SW(data,label,clusters):
    n = data.shape[1]
    S_W = np.zeros((n,n))
    mean_vectors = class_mean(data,label,clusters)
    for cl,mv in zip(range(1,clusters+1),mean_vectors):
        class_sc_mat = np.zeros((n,n))
        for row in data[label==cl]:
            row,mv = row.reshape(4,1),mv.reshape(4,1)
            class_sc_mat = class_sc_mat + (row-mv).dot((row-mv).T)
        S_W += class_sc_mat
    #print(S_W)
    return S_W

#计算类间散度矩阵，这里某一类的特征用改类的均值向量体现。 
#C个秩为1的矩阵的和，数据集中心是整体数据的中心，S_B是秩为C-1
def between_class_SB(data,label,clusters):
    n = data.shape[1]
    all_mean = np.mean(data,axis=0)
    S_B = np.zeros((n,n))
    mean_vectors = class_mean(data,label,clusters)
    for cl,mean_vec in enumerate(mean_vectors):
        ncl = data[label==cl+1,:].shape[0]
        mean_vec = mean_vec.reshape(4,1) # make column vector
        all_mean = all_mean.reshape(4,1)
        S_B += ncl*(mean_vec-all_mean).dot((mean_vec-all_mean).T)
    #print(S_B)
    return S_B

def lda():
    data,label = read_iris()
    clusters = 3
    S_W = within_class_SW(data,label,clusters)
    S_B = between_class_SB(data,label,clusters)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:,i].reshape(4,1)
        #print('---------------------')
        #print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
        #print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real)+'\n')
    eig_pairs = [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs,key=lambda k: k[0],reverse=True)
    print(eig_pairs)
    W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
    #print(W)
    #print(W.real)
    #print(data.dot(W))
    return W
    
def plot_lda():
    data,labels = read_iris()
    W = lda()
    print('-------------')
    print(data)
    print(W)
    print('------------------')
    Y = data.dot(W)
    print(Y)
    ax = plt.subplot(111)
    for label,marker,color in zip(range(1,4),('^','s','o'),('blue','red','green')):
        plt.scatter(x=Y[:,0][labels == label],
                    y=Y[:,1][labels == label],
                    marker = marker,
                    color = color,
                    alpha = 0.5
                    )
    plt.xlabel('lda1')
    plt.ylabel('lda2')
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')
    plt.show()
        
def default_plot_lda():
    Y = sklearnLDA()
    data,labels = read_iris()
    ax = plt.subplot(111)
    for label,marker,color in zip(range(1,4),('^','s','o'),('blue','red','green')):
        plt.scatter(x=Y[:,0][labels == label],
                    y=Y[:,1][labels == label],
                    marker = marker,
                    color = color,
                    alpha = 0.5
                    )
    plt.xlabel('LDA1')
    plt.ylabel('LDA2')
    plt.title('LDA:DEFAULT')
    plt.show()
    return Y
    
def sklearnLDA():
    from sklearn import datasets
    from sklearn.lda import LDA
    
    iris = datasets.load_iris()
    
    X=iris.data
    y=iris.target
    target_names = iris.target_names
    
    lda = LDA(n_components=2)
    X_r2 = lda.fit(X,y).transform(X)
    return X_r2

    
    


if __name__ == '__main__':
    plot_lda()
    default_plot_lda()
