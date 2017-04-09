'''
Deep Learning Programming Assignment 2
--------------------------------------
Name:
Roll No.:

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import time
import numpy as np
from matplotlib import pyplot as plt  
import pandas as pd
import theano
from theano import function, In, shared
from theano import tensor as T

def train(trainX, trainY):
        #initialise random weights
    #network architecture details
    L = 3
    input_units = 784
    hidden_units = 625
    output_units = 10
    learning_rate = 0.01
    training_steps = 100
    print 'Initialising weights...'
    #symbolic variables
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    W1_vals = np.asarray(rng.randn(input_units, hidden_units), dtype=theano.config.floatX)
    W1 = shared(value=W1_vals, name='W1')
    b1 = shared(value=rng.randn(hidden_units, ), name='b1')

    W2_vals = np.asarray(rng.randn(hidden_units, output_units), dtype=theano.config.floatX)
    W2 = shared(value=W2_vals, name='W2')
    b2 = shared(value=rng.randn(output_units, ), name='b2')


    #feed forward activations
    hidden_activations = T.nnet.sigmoid(T.dot(x, W1))
    prob_y_given_x = T.nnet.sigmoid(T.dot(hidden_activations, W2))
    predicted_idx = T.argmax(prob_y_given_x, axis=1)

    #cost 
    cost = T.mean(T.nnet.categorical_crossentropy(prob_y_given_x, y))
    params = [W1, W2]
    gradients = T.grad(cost, params)
    updates = [(param, param - learning_rate * grad) for param, grad in zip(params, gradients)]


    #compile functions
    print 'Compiling functions to train and predict...'
    train = function(inputs=[x, y], outputs=cost, updates=updates)
    predict = function(inputs=[x], outputs=[prob_y_given_x, predicted_idx])


    #train the model
    for i in range(training_steps):
        _cost = train(trainX, trainY)
        print 'Iteration: ', i+1, ' Cost: ', _cost
    np.savetxt('W1_weights.csv',W1, delimiter=",")
    np.savetxt('W2_weights.csv', W2, delimiter=",")
    np.savetxt('b1_weights.csv', b1, delimiter=",")
    np.savetxt('b2_weights.csv', b2, delimiter=",")


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
    W1 = np.genfromtxt('W1_weights.csv', delimiter=",")
    W2 = np.genfromtxt('W2_weights.csv', delimiter=",")
    b1 = np.genfromtxt('b1_weights.csv', delimiter=",")
    b2 = np.genfromtxt('b2_weights.csv', delimiter=",")
    count = 0
    for i in range(len(testX)):
        X = dataX[i:i+1]
        T = dataY[i:i+1]
        predict = function(inputs=testX, outputs=[prob_y_given_x, predicted_idx])
        
    return predict


