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
                z_1[z_1<=0] = 0                      # pass through ReLU non-linearity
                a_1 = z_1

                # pass trough the hidden layers 1
                z_2 = self.w_2.dot(a_1) + self.b_2
                z_2[z_2<=0] = 0
                a_2 = z_2                  

                # pass though the hidden layer 2
                z_3 = self.w_3.dot(a_2) + self.b_3
                # Activation function
                a_3 = self.__sigmoid(z_3) # predict class probabilities with the softmax activation function
                # Loss
                Yh = a_3
                
                loss = self.__squared_loss(Yh, Y_train)
                cost[i] += loss
               
                
                dLoss_Yh = Yh - Y_train

                dLoss_z3 = dLoss_Yh * self.__dev_sigmoid(z_3)
                dLoss_a2 = np.dot(self.w_3.T, dLoss_z3)
                dLoss_w3 = 1./a_2.shape[0] * np.dot(dLoss_z3, a_2.T)
                dLoss_b3 = 1./a_2.shape[0] * np.dot(dLoss_z3, np.ones([dLoss_z3.shape[1],1]))
                
#                 print("dLoss_b3",dLoss_b3.shape)
#                 print("dLoss_w3",dLoss_w3.shape)
#                 print("a_2.shape",a_2)
#                 print("dLoss_z3",dLoss_z3)
                
                # 2nd layer
                dLoss_z2 = dLoss_a2 * self.__relu_derivative(z_2)        
                dLoss_a1 = np.dot(self.w_2.T,dLoss_z2)
                dLoss_w2 = 1./a_1.shape[1] * np.dot(dLoss_z2, a_1.T)
                dLoss_b2 = 1./a_1.shape[1] * np.dot(dLoss_z2, np.ones([dLoss_z2.shape[1],1]))
                
#                 print("1./a_1.shape[1]",(1./a_1.shape[0]))
#                 print("dLoss_w2",dLoss_w2)
                
                
                # 1st layer
                dLoss_z1 = dLoss_a1 * self.__relu_derivative(z_1)        
                dLoss_a0 = np.dot(self.w_1.T,dLoss_z1)
                dLoss_w1 = 1./element.shape[1] * np.dot(dLoss_z1, element.T)
                dLoss_b1 = 1./element.shape[1] * np.dot(dLoss_z1, np.ones([dLoss_z1.shape[1],1]))
                
#                 print("dLoss_b1",dLoss_b1.shape)
#                 print("dLoss_w1",dLoss_w1.shape)
                
                # Update the weight and biases

                self.w_1 = self.w_1 - dLoss_w1 * alpha
                self.b_1 = self.b_1 - dLoss_b1 * alpha
                self.w_2 = self.w_2 - dLoss_w2 * alpha
                self.b_2 = self.b_2 - dLoss_b2 * alpha
                self.w_3 = self.w_3 - dLoss_w3 * alpha
                self.b_3 = self.b_3 - dLoss_b3 * alpha
                
                params = [self.w_1, self.b_1, self.w_2, self.b_2, self.w_3, self.b_3]
                # print("params",params[3:4])

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
        z1[z1<=0] = 0 # pass through ReLU non-linearity

        z2 = w2.dot(z1) + b2 # first dense layer
        z2[z2<=0] = 0 # pass through ReLU non-linearity

        out = w3.dot(z2) + b3 # second dense layer
        probs = self.__sigmoid(out) # predict class probabilities with the softmax activation function

        return (np.argmax(probs), np.max(probs))
    
    #### private methods
    
    def __squared_loss(self, probs, labels):
        return np.sum((probs - labels)**2)
    
    def __relu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def __sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    
    def __dev_sigmoid(self, Z):
        s = 1/(1+np.exp(-Z))
        dZ = s * (1-s)
        return dZ

