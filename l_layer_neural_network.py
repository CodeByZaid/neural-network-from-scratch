import numpy as np
import matplotlib.pyplot as plt
import copy

def initialize_parameters_deep(layer_dims):     
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters ['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters ['b' + str(l)] = np.zeros((layer_dims[l],1))  
       
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters

def sigmoid(Z):
    
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def relu(Z):
    
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dA to a correct object.
    dZ[Z <= 0] = 0
    return dZ

def linear_forward(A, W, b):   
    
    Z = np.dot(W,A)+b
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
        
    if activation == "sigmoid":
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    
    elif activation == "relu":
        
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache=relu(Z)        
    
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    
    caches = []
    A = X
    L = len(parameters) // 2                # number of layers in the neural network   
    
    for l in range(1, L):
        A_prev = A 
        
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)       
        
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)],parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)      
          
    return AL, caches

def compute_cost(AL, Y):
    
    m = Y.shape[1]    
    cost = (-1/m) * np.sum(np.multiply(Y,np.log(AL)) + np.multiply(1-Y,np.log(1-AL)))   
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
      
    return cost

def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)   
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":        
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
                
    elif activation == "sigmoid":        
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
            
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
   
    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp    
    
    for l in reversed(range(L-1)):
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp     
    

    return grads

def update_parameters(params, grads, learning_rate):
    
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2 # number of layers in the neural network
   
    for l in range(L):
        
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
        
    return parameters

def predict(X, parameters, threshold=0.5):
    AL, _ = L_model_forward(X, parameters)
    predictions = (AL > threshold).astype(int)
    return predictions

def accuracy(Y, Y_hat):
    return np.mean(Y_hat == Y)

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False,plot_cost=True):
    
    np.random.seed(1)
    costs = []                         # keep track of cost
    parameters = initialize_parameters_deep(layers_dims)
    
    
    for i in range(0, num_iterations):        
        AL, caches = L_model_forward(X, parameters)      
        cost = compute_cost(AL, Y)       
        grads = L_model_backward(AL, Y, caches)        
        parameters = update_parameters(parameters, grads, learning_rate)        
        
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0:
            costs.append(cost)
    if plot_cost:
        plt.plot(np.arange(0, len(costs) * 100, 100), costs)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Training cost")
        plt.show()
    return parameters, costs


#Quick testing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, Y = make_moons(n_samples=500, noise=0.2, random_state=3)
Y = Y.reshape(1, -1)
X = X.T

X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size=0.2, random_state=42)
X_train, X_test = X_train.T, X_test.T
Y_train, Y_test = Y_train.T, Y_test.T

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

layers_dims = [X_train.shape[0], 10, 5, 1]

parameters, costs = L_layer_model(X_train, Y_train,
                                  layers_dims,
                                  learning_rate=0.05,
                                  num_iterations=2000,
                                  print_cost=True)

Y_pred_train = predict(X_train, parameters)
Y_pred_test = predict(X_test, parameters)

print("Train accuracy:", np.mean(Y_pred_train == Y_train) * 100, "%")
print("Test accuracy:", np.mean(Y_pred_test == Y_test) * 100, "%")



