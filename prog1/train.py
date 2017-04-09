'''
Deep Learning Programming Assignment 1
--------------------------------------
Name:Rohan Khameshra
Roll No.:15EE10037

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import math
import os
import scipy as sp
import pandas as pd
import sklearn


def sigmoid(X,W,b):
    """
    Computes H = sigmoid(X . W + b) 
    
    """
    pre = np.dot(X, W) + b
    return (1.0)/(1.0 + np.exp(-pre))


def Gradient(sigma, Y, T, V, X):
    V_gradient = np.dot(sigma.T, Y-T)/sigma.shape[0]
    d_gradient = (Y-T).mean(axis=0)
    W_gradient = np.dot(X.T, np.dot(Y-T, V.T)*sigma*(1-sigma))/X.shape[0]
    b_gradient = (np.dot((Y-T), V.T)) * sigma.T* ((1 - sigma))
    b_gradient = b_gradient.mean(axis=0)
    return [V_gradient, d_gradient, W_gradient, b_gradient,]



def softmax(a, b, c):
        
    """
    a: Batch of hidden unit activations of shape (batch_size, num_hid)
    b :Weight matrix of shape (num_hid, num_classes)
    c : Bias vector of shape (num_classes, )
    """
    post = np.dot(a,b) + c
    exVector = np.exp(post)
    return exVector/(np.sum(exVector, axis=1)[:,np.newaxis])

def cross(Y, T):
    """
    This function returns cross entropy loss 
    """
    loss = -(T*np.log(Y)).sum(axis=1).mean(axis=0)
    return loss

def Weightupdate(V, d, W, b, learning_rate, grad_L):
    V -= learning_rate*grad_L[0]
    d -= learning_rate*grad_L[1]
    W -= learning_rate*grad_L[2]
    b -= learning_rate*grad_L[3]
    return [V, d, W, b]

def fwdprop(X,W,b,V,d):
    sigma= sigmoid(X, W, b)
    Y = softmax(sigma, V, d)
    return sigma, Y



def train(trainX, trainY):
    '''
    Complete this function.
    '''
    inputSize = 28*28
    hiddenSize = 100
    noClasses = 10
    W = -0.01*np.random.randn(inputSize,hiddenSize)
    b = np.zeros([hiddenSize])
    V = -0.01*np.random.randn(hiddenSize, noClasses)
    d = np.zeros([noClasses])
    
    epochs = 10
    batchSize = 200
    count = 0
    learning_rate = 0.01
    noClasses = 10    
    for i in range(epochs) :
        for j in range(int(len(trainX)/batchSize)) :
            X = trainX[j*batchSize:(j+1)*batchSize]
            T = trainY[j*batchSize:(j+1)*batchSize]
            k = np.zeros((T.size, noClasses))
            k[np.arange(T.size), T] = 1
            T = k 
            [H, Y] = fwdprop(X, W, b, V, d)
            lossValue = cross(Y, T)
            if count%100 == 0 :
                print (lossValue)
            grad_List = Gradient(H, Y, T, V, X)
#                 for i in range(len(gradList)):
#                     print gradList[i].shape
            [V, d, W, b] = Weightupdate(V, d, W, b, learning_rate, grad_List)
            count+=1
    np.savetxt('V_weights.csv',V, delimiter=",")
    np.savetxt('d_weights.csv', d, delimiter=",")
    np.savetxt('W_weights.csv', W, delimiter=",")
    np.savetxt('b_weights.csv', b, delimiter=",")
    



def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''
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

    return np.zeros(testX.shape[0])
