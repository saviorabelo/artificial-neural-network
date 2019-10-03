import math
import random
import numpy as np
from Utils.utils import Util as util
from matplotlib import pyplot as plt
from pandas_ml import ConfusionMatrix

class Perceptron:
    def __init__(self, x_data, y_data, activation):
        self.x_data = x_data
        if activation == 'tanh':
            self.y_data = np.where(y_data == 1, 1, -1)
        else:
            self.y_data = y_data
        self.attributes = x_data.shape[1]
        self.output_layer = y_data.shape[1]
        self.epochs = 200
        self.realizations = 10
        self.train_size = 0.8
        self.activation = activation
        self.hit_rate = []
        self.tpr = []
        self.spc = []
        self.ppv = []
    
    def initWeigths(self):
        self.w = np.random.rand(self.output_layer, self.attributes+1)

    def activationFunction(self, u):
        if self.activation == 'step':
            value = np.nanmax(u)
            y = np.where(u == value, 1, 0)
            y_ = 1
            return y, y_
        elif self.activation == 'logistic':
            u = 1.0/(1.0 + np.exp(-u))
            value = np.nanmax(u)
            y = np.where(u == value, 1, 0)
            y_ = u * (1.0 - u)
            return y, y_
        elif self.activation == 'tanh':
            u = (np.exp(u) - np.exp(-u))/(np.exp(u) + np.exp(-u))
            value = np.nanmax(u)
            y = np.where(u == value, 1, -1)
            y_ = 0.5 * (1.0 - (u * u))
            return y, y_
        else:
            print('Error in activation function!\n')

    def predict(self, xi):
        u = np.dot(self.w, xi)
        y, y_ = self.activationFunction(u)
        return y, y_
    
    def updateEta(self, epoch):
        eta_i = 0.05
        eta_f = 0.5
        eta = eta_f * ((eta_i / eta_f) ** (epoch / self.epochs))
        self.eta = eta

    def train(self, x_train, y_train):
        stop_error = 1
        cont_epochs = 0
        vector_error = []
        while (stop_error and cont_epochs < self.epochs):
            self.updateEta(cont_epochs)
            stop_error = 0
            x_train, y_train = util.shuffleData(x_train, y_train)
            (m, _) = x_train.shape
            for i in range(m):
                xi = x_train[i]
                y, y_ = self.predict(xi)
                d = y_train[i]
                error = d - y

                # check if this error
                if not np.array_equal(error, [0, 0, 0]):
                    stop_error = 1

                # update weights
                aux = (y_ * error).reshape(-1, 1)
                self.w += self.eta * (aux * xi)
            cont_epochs += 1

    def test(self, x_test, y_test):
        (m, _) = x_test.shape
        y_actu = []
        y_pred = []
        for i in range(m):
            xi = x_test[i]
            y, _ = self.predict(xi)
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
