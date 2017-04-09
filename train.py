'''
Deep Learning Programming Assignment 1
--------------------------------------
Name:
Roll No.:

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
from scipy.optimize import minimize

def sigmoid(X):
    return 1.0 / (1.0 + math.e ** (-1.0 * X)) 

# Randomly initializes the weights for layer with the specified numbers of
# incoming and outgoing connections.
def randInitializeWeights(incoming, outgoing):
    epsilon_init = 0.12
    return rand(outgoing, 1 + incoming) * (2 * epsilon_init) - epsilon_init

# Adds the bias column to the matrix X.
def addBias(X):
    return np.concatenate((np.ones((X.shape[0],1)), X), 1) 

# Reconstitutes the two weight matrices from a single vector, given the
# size of the input layer, the hidden layer, and the number of possible
# labels in the output.
def extractWeightMatrices(thetas, input_layer_size, hidden_layer_size, num_labels):
    theta1size = (input_layer_size + 1) * hidden_layer_size
    theta1 = reshape(thetas[:theta1size], (hidden_layer_size, input_layer_size + 1), order='A')
    theta2 = reshape(thetas[theta1size:], (num_labels, hidden_layer_size + 1), order='A')
    return theta1, theta2

# Converts single lables to one-hot vectors.
def convertLabelsToClassVectors(labels, num_classes):
    labels = labels.reshape((labels.shape[0],1))
    ycols = np.tile(labels, (1, num_classes))
    m, n = ycols.shape
    indices = np.tile(np.arange(num_classes).reshape((1,num_classes)), (m, 1))
    ymat = indices == ycols
    return ymat.astype(int)

# Returns a vector corresponding to the randomly initialized weights for the
# input layer and hidden layer.
def getInitialWeights(input_layer_size, hidden_layer_size, num_labels):
    theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    theta2 = randInitializeWeights(hidden_layer_size, num_labels)
    return np.append(theta1.ravel(order='A'), theta2.ravel(order='A'))

# Trains a basic multilayer perceptron. Returns weights to use for feed-forward
# pass to predict on new data.
def train_nn(X_train, y_train, hidden_layer_size, lmda, maxIter):
    input_layer_size = X_train.shape[1]
    num_labels = 10
    initial_weights = getInitialWeights(input_layer_size, hidden_layer_size, num_labels)
    if y_train.ndim == 1:
        # Convert the labels to one-hot vectors.
        y_train = convertLabelsToClassVectors(y_train, num_labels)

    # Given weights for the input layer and hidden layer, calulates the 
    # activations for the hidden layer and the output layer of a 3-layer nn.
    def getActivations(theta1, theta2):
        z2 = np.dot(addBias(X_train),theta1.T)
        a2 = np.concatenate((np.ones((z2.shape[0],1)), sigmoid(z2)), 1)
        # a2 is an m x num_hidden+1 matrix, Theta2 is a num_labels x
        # num_hidden+1 matrix
        z3 = np.dot(a2,theta2.T)
        a3 = sigmoid(z3) # Now we have an m x num_labels matrix
        return a2, a3

    # Cost function to be minimized with respect to weights.
    def costFunction(weights):
        theta1, theta2 = extractWeightMatrices(weights, input_layer_size, hidden_layer_size, num_labels)
        hidden_activation, output_activation = getActivations(theta1, theta2)
        m = X_train.shape[0]
        cost = sum((-y_train * log(output_activation)) - ((1 - y_train) * log(1-output_activation))) / m
        # Regularization
        thetasq = sum(theta1[:,1:(input_layer_size + 1)]**2) + sum(theta2[:,1:hidden_layer_size + 1]**2)
        reg = (lmda / float(2*m)) * thetasq
        print("Training loss:\t\t{:.6f}".format(cost))
        return cost + reg

    # Gradient function to pass to our optimization function.
    def calculateGradient(weights):
        theta1, theta2 = extractWeightMatrices(weights, input_layer_size, hidden_layer_size, num_labels)
        # Backpropagation - step 1: feed-forward.
        hidden_activation, output_activation = getActivations(theta1, theta2)
        m = X_train.shape[0]
        # Step 2 - the error in the output layer is just the difference
        # between the output layer and y
        delta_3 = output_activation - y_train # delta_3 is m x num_labels
        delta_3 = delta_3.T

        # Step 3
        sigmoidGrad = hidden_activation * (1 - hidden_activation)
        delta_2 = (np.dot(theta2.T,delta_3)) * sigmoidGrad.T
        delta_2 = delta_2[1:, :] # hidden_layer_size x m
        theta1_grad = np.dot(delta_2, np.concatenate((np.ones((X_train.shape[0],1)), X_train), 1))
        theta2_grad = np.dot(delta_3, hidden_activation)
        # Add regularization
        reg_grad1 = (lmda / float(m)) * theta1
        # We don't regularize the weight for the bias column
        reg_grad1[:,0] = 0
        reg_grad2 = (lmda / float(m)) * theta2;
        reg_grad2[:,0] = 0
        return np.append(ravel((theta1_grad / float(m)) + reg_grad1, order='A'), ravel((theta2_grad / float(m)) + reg_grad2, order='A'))

    # Use scipy's minimize function with method "BFGS" to find the optimum
    # weights.
    res = minimize(costFunction, initial_weights, method='BFGS', jac=calculateGradient, options={'disp': False, 'maxiter':maxIter})
    theta1, theta2 = extractWeightMatrices(res.x, input_layer_size, hidden_layer_size, num_labels)
    return theta1, theta2

# Predicts the output given input and weights.
def predict(X, theta1, theta2):
    m, n = X.shape
    X = addBias(X)
    h1 = sigmoid(np.dot(X,theta1.T))
    h2 = sigmoid(addBias(h1).dot(theta2.T))
    return np.argmax(h2, axis=1)


def train(trainX, trainY):
    '''
    Complete this function.
    '''
    init_weights = getInitialWeights(784, 1, 10)
    theta1_init, theta2_init = extractWeightMatrices(init_weights, 784, 1, 10)
    theta1, theta2 = train_nn(X_train, y_train, 1, 0, 50)
    predictions = predict(X_train, theta1, theta2)

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
    predictions = predict(X_test, theta1, theta2)

    return np.zeros(testX.shape[0])
