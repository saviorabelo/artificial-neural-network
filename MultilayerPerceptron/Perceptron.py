import math
import random
import numpy as np
from Utils.utils import Util as util
from matplotlib import pyplot as plt
from pandas_ml import ConfusionMatrix

class Perceptron:
    def __init__(self, x_data, y_data, activation, hidden_layer):
        self.x_data = x_data
        if activation == 'tanh':
            self.y_data = np.where(y_data == 1, 1, -1)
        else:
            self.y_data = y_data
        self.attributes = x_data.shape[1]
        self.hidden_layer = hidden_layer
        self.output_layer = y_data.shape[1]
        self.epochs = 150
        self.realizations = 1
        self.precision = 10**(-5)
        self.train_size = 0.8
        self.activation = activation
        self.hit_rate = []
        self.tpr = []
        self.spc = []
        self.ppv = []
    
    def initWeigths(self):
        self.w = np.random.random((self.attributes+1, self.hidden_layer))
        self.m = np.random.random((self.hidden_layer+1, self.output_layer))

    def function(self, u):
        if self.activation == 'step':
            y = u
        elif self.activation == 'logistic':
            y = 1.0/(1.0 + np.exp(-u))
        elif self.activation == 'tanh':
            y = (np.exp(u) - np.exp(-u))/(np.exp(u) + np.exp(-u))
        return y    
    
    def derivate(self, u):
        if self.activation == 'step':
            y_ = 1
        elif self.activation == 'logistic':
            y_ = u * (1.0 - u)
        elif self.activation == 'tanh':
            y_ = 0.5 * (1.0 - (u * u))
        return y_
    
    def activationFunction(self, u):
        value = np.nanmax(u)
        if self.activation == 'step' or self.activation == 'logistic':
            y = np.where(u == value, 1, 0)
        elif self.activation == 'tanh':
            y = np.where(u == value, 1, -1)
        return y

    def predict(self, xi):
        u = np.dot(self.w, xi)
        u, y, y_ = self.activationFunction(u)
        return u, y, y_
    
    def updateEta(self, epoch):
        eta_i = 0.05
        eta_f = 0.5
        eta = eta_f * ((eta_i / eta_f) ** (epoch / self.epochs))
        self.eta = eta

    def train(self, x_train, y_train):
        error_old = 0
        cont_epochs = 0
        mse_vector = []
        while True:
            self.updateEta(cont_epochs)
            x_train, y_train = util.shuffleData(x_train, y_train)
            (m, _) = x_train.shape
            error_epoch = 0
            for i in range(m):
                xi = x_train[i]
                H = np.dot(xi, self.w)
                H = self.function(H)
                H_ = self.derivate(H)
                
                H = np.concatenate(([-1], H), axis=None)
                Y = np.dot(H, self.m)
                Y = self.function(Y)
                Y_ = self.derivate(Y)

                # Quadratic Error Calculation
                d = y_train[i]
                error = d - Y
                error_epoch += np.sum(error**2)
                
                # Output layer
                delta_output = (error * Y_).reshape(-1, 1)
                aux = (self.eta * delta_output)
                self.m += np.dot(H.reshape(-1, 1), aux.T)

                # Hidden layer
                delta_hidden = np.sum(np.dot(self.m, delta_output)) * H_
                aux = (self.eta * delta_hidden).reshape(-1, 1)
                self.w += np.dot(xi.reshape(-1, 1), aux.T)

            mse = error_epoch / m
            mse_vector.append(mse)

            if abs(error_epoch - error_old) <= self.precision:
                #print('Stop Precision: {}'.format(abs(error_epoch - error_old)))
                break
            if cont_epochs >= self.epochs:
                #print('Stop Epochs: {}'.format(cont_epochs))
                break
        
            error_old = error_epoch
            cont_epochs += 1
        #util.plotErrors(mse_vector)

    def test(self, x_test, y_test):
        (m, _) = x_test.shape
        y_actu = []
        y_pred = []
        for i in range(m):

            xi = x_test[i]
            H = np.dot(xi, self.w)
            H = self.function(H)

            H = np.concatenate(([-1], H), axis=None)
            Y = np.dot(H, self.m)
            Y = self.function(Y)
            y = self.activationFunction(Y)
            d = y_test[i]
           
            # Confusion Matrix
            y_actu.append(list(d))
            y_pred.append(list(y))

        a = util.inverse_transform(y_actu)
        b = util.inverse_transform(y_pred)
        cm = ConfusionMatrix(a, b)
        #cm.print_stats()
        #util.plotConfusionMatrix(cm)

        return cm.ACC, cm.TPR, cm.SPC, cm.PPV 

    def execute(self):
        # Pre processing
        x_data = util.normalizeData(self.x_data)
        x_data = util.insertBias(x_data)
        y_data = self.y_data

        for i in range(self.realizations):
            self.initWeigths()
            x_data_aux, y_data_aux = util.shuffleData(x_data, y_data)
            x_train, x_test, y_train, y_test = util.splitData(x_data_aux, y_data_aux, self.train_size)
            self.train(x_train, y_train)
            acc, tpr, spc, ppv = self.test(x_test, y_test)
            
            self.hit_rate.append(acc)
            self.tpr.append(tpr)
            self.spc.append(spc)
            self.ppv.append(ppv)
        
        #util.plotColorMap(x_train, x_test, y_train, self.predict)

        self.acc = np.mean(self.hit_rate)
        self.std = np.std(self.hit_rate)
        self.tpr = np.mean(self.tpr)
        self.spc = np.mean(self.spc)
        self.ppv = np.mean(self.ppv)

        print('Hit rate: {}'.format(self.hit_rate))
        print('Accuracy: {:.2f}'.format(self.acc*100))
        print('Minimum: {:.2f}'.format(np.amin(self.hit_rate)*100))
        print('Maximum: {:.2f}'.format(np.amax(self.hit_rate)*100))
        print('Standard Deviation: {:.2f}'.format(self.std))
        print('Sensitivity: {:.2f}'.format(self.tpr*100))
        print('Specificity: {:.2f}'.format(self.spc*100))
        print('Precision: {:.2f}'.format(self.ppv*100))
