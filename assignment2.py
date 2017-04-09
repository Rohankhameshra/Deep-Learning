
# coding: utf-8

# In[1]:

'''
Deep Learning Programming Assignment 1
--------------------------------------
Name: Rajat Bhavnani
Roll No.: 13EC10048


======================================

Problem Statement:
Implement a simple 1 hidden layer MLP WITHOUT using any deep learning library
for predicting MNIST images. You are allowed to use linear algebra
libraries like numpy.

Resources:
1. https://ift6266h16.wordpress.com/2016/01/11/first-assignment-mlp-on-mnist/
2. https://github.com/tfjgeorge/ift6266/blob/master/notebooks/MLP.ipynb
    (In french. But the same repository has other useful ipython notebooks)

You might want to first code in an ipython notebook and later copy-paste
the code here.



======================================

Instructions:
1. Download the MNIST dataset from http://yann.lecun.com/exdb/mnist/
    (four files).
2. Extract all the files into a folder named `data' just outside
    the folder containing the main.py file. This code reads the
    data files from the folder '../data'.
3. Complete the functions in the train.py file. You might also
    create other functions for your convenience, but do not change anything
    in the main.py file or the function signatures of the train and test
    functions in the train.py file.
4. The train function must train the neural network given the training
    examples and save the in a folder named `weights' in the same
    folder as main.py
5. The test function must read the saved weights and given the test
    examples it must return the predicted labels.
6. Submit your project folder with the weights. Note: Don't include the
    data folder, which is anyway outside your project folder.

Submission Instructions:
1. Fill your name and roll no in the space provided above.
2. Name your folder in format <Roll No>_<First Name>.
    For example 12CS10001_Rohan
3. Submit a zipped format of the file (.zip only).
'''

import numpy as np
import math
import os
import scipy as sp
import pandas as pd
import sklearn



# In[32]:


def softmax(a, b, c):
        
    """
    To compute output=softmax(a .b +c) corresponding to the output probabilities
    of a one-hidden-layer multi-layer perceptron classifier where 
    a: Batch of hidden unit activations of shape (batch_size, num_hid)
    b :Weight matrix of shape (num_hid, num_classes)
    c : Bias vector of shape (num_classes, )
    """
    post = np.dot(a,b) + c
    exVector = np.exp(post)
    return exVector/(np.sum(exVector, axis=1)[:,np.newaxis])


def sigmoid(X,W,b):
    """
    Computes H = sigmoid(X . W + b) corresponding to the hidden unit
    activations of a one-hidden-layer MLP classifier
    
    """
    pre = np.dot(X, W) + b
    return (1.0)/(1.0 + np.exp(-pre))



def CEL(Y, T):
    """
    This function returns teh binary cross entropy loss of the classifier
    
    """
    loss = -(T*np.log(Y)).sum(axis=1).mean(axis=0)
    return loss

############### TEST LOSS #######################

# V = np.random.rand(4,10)
# print V
# k = np.random.randint(9, size=4)
# print k
# d = np.zeros((k.size,10), dtype=np.int)
# d[np.arange(k.size), k] = 1
# print d

# loss(V,d)

def fwdprop(X,W,b,V,d):
    """
    this function performs forward propagation on the multi-level perceptron classifier
    Parameters
    ----------
    X : numpy.ndarray
        Batch of examples of shape (batch_size, num_vis)
    W : numpy.ndarray
        Weight matrix of shape (num_vis, num_hid)
    b : numpy.ndarray
        Bias vector of shape (num_hid, )
    V : numpy.ndarray
        Weight matrix of shape (num_hid, num_classes)
    d : numpy.ndarray
        Bias vector of shape (num_classes, )
    Returns
    -------
sigma : Batch of activations in hidden layer shape (batch_size, num_hid)
    Y : Batch of probability vectors of shape (batch_size, num_classes)
    """
    sigma= sigmoid(X, W, b)
    Y = softmax(sigma, V, d)
    return sigma, Y

def Gradient(sigma, Y, T, V, X):
    V_gradient = np.dot(sigma.T, Y-T)/sigma.shape[0]
    d_gradient = (Y-T).mean(axis=0)
#     print Y-T
#     print dGrad
#     print "dGrad.shape", dGrad.shape
    W_gradient = np.dot(X.T, np.dot(Y-T, V.T)*sigma*(1-sigma))/X.shape[0]
    b_gradient = (np.dot((Y-T), V.T)) * sigma.T* ((1 - sigma))
#     print bGrad
#     print bGrad.shape
#     k = bGrad.mean(axis=0)
#     print "k",k.shape
    b_gradient = b_gradient.mean(axis=0)
    return [V_gradient, d_gradient, W_gradient, b_gradient,]


################ Gradient #######################

# X = np.random.randint(5, size=[2,4])
# print X, "X"
# H = np.random.randint(5, size=[2,4])
# print "H", H
# Y = np.random.randint(5, size=[2,10])
# print "Y", Y
# k = np.random.randint(9, size=2)
# print "k", k
# T = np.zeros((k.size,10), dtype=np.int)
# T[np.arange(k.size), k] = 1
# print "T", T
# V = np.random.randint(5, size=[4,10])
# print "V", V
# print calculateGradient(H,Y,T)



def Weightupdate(V, d, W, b, learnRate, grad_L):
    V -= learnRate*grad_L[0]
    d -= learnRate*grad_L[1]
    W -= learnRate*grad_L[2]
    b -= learnRate*grad_L[3]
    return [V, d, W, b]

def train(W, b, V, d, dataX,dataY):
        
        ############# Model Parameters #########################
        
        epochs = 5
        batchSize = 100
        count = 0
        learnRate = 0.001
        noClasses = 10
                 
        
        ################### Model Training #########################
        
        for i in range(epochs) :
            for j in range(int(len(dataX)/batchSize)) :
                X = dataX[j*batchSize:(j+1)*batchSize]
                T = dataY[j*batchSize:(j+1)*batchSize]
                k = np.zeros((T.size, noClasses))
                k[np.arange(T.size), T] = 1
                T = k 
                [H, Y] = fwdprop(X, W, b, V, d)
                lossValue = CEL(Y, T)
                if count%100 == 0 :
                    print (lossValue)
                grad_List = Gradient(H, Y, T, V, X)
#                 for i in range(len(gradList)):
#                     print gradList[i].shape
                [V, d, W, b] = Weightupdate(V, d, W, b, learnRate, grad_List)
                count+=1
        np.savetxt('V_weights.csv',V, delimiter=",")
        np.savetxt('d_weights.csv', d, delimiter=",")
        np.savetxt('W_weights.csv', W, delimiter=",")
        np.savetxt('b_weights.csv', b, delimiter=",")
        

def test(dataX, dataY):
    V = np.genfromtxt('V_weights.csv', delimiter=",")
    d = np.genfromtxt('d_weights.csv', delimiter=",")
    W = np.genfromtxt('W_weights.csv', delimiter=",")
    b = np.genfromtxt('b_weights.csv', delimiter=",")
    count = 0
    for i in range(len(dataX)):
        X = dataX[i:i+1]
        T = dataY[i:i+1]
        [H, Y] = fwdprop(X, W, b, V, d)
        lossValue = CEL(Y, T)
        Y = np.argmax(Y)
#         print Y, T
        if(Y==T):
            count += 1
    print ("Accuracy::  %g" %(float(count)/float(len(dataX))))
    return 




# In[33]:

def load_mnist():
    data_dir = '../data'
#     print os.path.join(data_dir, 'train-images.idx3-ubyte')
    fd = open(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int)

    fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    perm = np.random.permutation(trY.shape[0])
    trX = trX[perm]
    trY = trY[perm]

    perm = np.random.permutation(teY.shape[0])
    teX = teX[perm]
    teY = teY[perm]

    return trX, trY, teX, teY

def main():
    trainX, trainY, testX, testY = load_mnist()
    trainX = np.reshape(trainX,[-1,784])
    testX = np.reshape(testX,[-1,784])
    print (trainX.shape)
    print (trainY.shape)
    #print testX.shape
    
    ############## Model Weight Vectors #####################
        
    inputSize = 28*28
    hiddenSize = 100
    noClasses = 10

    ############## Training Data ###########################3
    
    W = -0.01*np.random.randn(inputSize,hiddenSize)
    b = np.zeros([hiddenSize])
    V = -0.01*np.random.randn(hiddenSize, noClasses)
    d = np.zeros([noClasses])
    
    train(W, b, V, d, trainX, trainY)
    test(testX, testY)   
    
    
if __name__ == '__main__':
    main()    


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



