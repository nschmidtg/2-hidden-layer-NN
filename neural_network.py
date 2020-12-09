import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd

class NNetwork:
    def __init__(self, n_input, hidden_layer_size, n_cat):
        self.n_input = n_input
        self.n_hidden_layers = 2
        self.hidden_layer_size = hidden_layer_size
        self.n_cat = n_cat
        
        # initialize weights
        self.w_1 = np.random.randn(self.hidden_layer_size, self.n_input) / np.sqrt(self.n_input)
        self.w_2 = np.random.randn(self.hidden_layer_size, self.hidden_layer_size) / np.sqrt(self.hidden_layer_size)
        self.w_3 = np.random.randn(self.n_cat, self.hidden_layer_size) / np.sqrt(self.hidden_layer_size)

        # initialize biases
        self.b_1 = np.zeros((self.hidden_layer_size, 1))
        self.b_2 = np.zeros((self.hidden_layer_size, 1))
        self.b_3 = np.zeros((self.n_cat, 1))
        
    def train(self, trainset_i, n_epochs, alpha, n_samples):
        """
        train a fully connected neural network using the specified train_set (dataframe)
        """
        if (trainset_i).shape[1]-1 != self.n_input:
            raise ValueError(u"X_train size has to be the same as the Network inputs")

        i = 0
        cost = np.zeros(n_epochs)

        while i < n_epochs:            
            subset = trainset_i.sample(n_samples)
            for index, element in trainset_i.iterrows():
                
                X_train = element[:-1]
                Y_train = element[self.n_input]
                dim1 = X_train.shape
                
                element = X_train.values.reshape((dim1[0], 1))
             
                z_1 = self.w_1.dot(element) + self.b_1    # input weight
                a_1 = self.__relu(z_1)                   # pass through ReLU non-linearity

                # pass trough the hidden layer 1
                z_2 = self.w_2.dot(a_1) + self.b_2
                a_2 = self.__relu(z_2)               

                # pass though the hidden layer 2
                z_3 = self.w_3.dot(a_2) + self.b_3
                # Activation function
                a_3 = self.__sigmoid(z_3) # predict class probabilities with the softmax activation function
                # Loss
                Yh = a_3
                
                
                loss = self.__squared_loss(Yh, Y_train)
                cost[i] += loss
               
                # derivative of the loss function w.r.t. output a_3
                dLoss_Yh = Yh - Y_train

                dLoss_z3 = dLoss_Yh * self.__dev_sigmoid(z_3)
                dLoss_a2 = np.dot(self.w_3.T, dLoss_z3)
                dLoss_w3 = 1./a_2.shape[0] * np.dot(dLoss_z3, a_2.T)
                dLoss_b3 = 1./a_2.shape[0] * np.dot(dLoss_z3, np.ones([dLoss_z3.shape[1],1]))
                
                # 2nd layer
                dLoss_z2 = dLoss_a2 * self.__relu_derivative(z_2)        
                dLoss_a1 = np.dot(self.w_2.T, dLoss_z2)
                dLoss_w2 = 1./a_1.shape[1] * np.dot(dLoss_z2, a_1.T)
                dLoss_b2 = 1./a_1.shape[1] * np.dot(dLoss_z2, np.ones([dLoss_z2.shape[1],1]))
                
                # 1st layer
                dLoss_z1 = dLoss_a1 * self.__relu_derivative(z_1)        
                dLoss_a0 = np.dot(self.w_1.T,dLoss_z1)
                dLoss_w1 = 1./element.shape[1] * np.dot(dLoss_z1, element.T)
                dLoss_b1 = 1./element.shape[1] * np.dot(dLoss_z1, np.ones([dLoss_z1.shape[1],1]))

                # Update the weight and biases
                self.w_1 = self.w_1 - dLoss_w1 * alpha
                self.b_1 = self.b_1 - dLoss_b1 * alpha
                self.w_2 = self.w_2 - dLoss_w2 * alpha
                self.b_2 = self.b_2 - dLoss_b2 * alpha
                self.w_3 = self.w_3 - dLoss_w3 * alpha
                self.b_3 = self.b_3 - dLoss_b3 * alpha
                
                params = [self.w_1, self.b_1, self.w_2, self.b_2, self.w_3, self.b_3]

            i += 1
        to_save = [params, cost/float(n_samples)]
    
        with open('model', 'wb') as file:
            pickle.dump(to_save, file)

        return cost
    
    def predict(self, element, w1, b1, w2, b2, w3, b3):
        '''
        Make predictions with trained filters/weights. 
        '''
        element = element[:-1]
        fc = element.values.reshape((60, 1)) # flatten pooled layer

        z1 = w1.dot(fc) + b1 # first dense layer
        a1 = self.__relu(z1) # pass through ReLU non-linearity

        z2 = w2.dot(a1) + b2 # first dense layer
        a2 = self.__relu(z2) # pass through ReLU non-linearity

        out = w3.dot(a2) + b3 # second dense layer
        probs = self.__sigmoid(out) # predict class probabilities with the softmax activation function

        return (np.argmax(probs), np.max(probs))
    
    #### private methods
    
    def __squared_loss(self, probs, labels):
        return np.sum((probs - labels)**2)*0.5
    
    def __relu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    
    def __relu(self, X):
        return np.maximum(0,X)

    def __sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    
    def __dev_sigmoid(self, Z):
        s = 1/(1+np.exp(-Z))
        dZ = s * (1-s)
        return dZ

