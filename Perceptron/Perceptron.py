import random
import numpy as np
from utils import *
from pandas_ml import ConfusionMatrix

class Perceptron:

    def __init__(self, x_data, y_data, activation='degree', realizations=5, \
            learning_rate=0.05, epochs=200, train_size=0.8):
        self.x_data = x_data
        self.y_data = y_data
        self.activation = activation
        self.attributes = x_data.shape[1]
        self.output_layer = y_data.shape[1]
        self.eta = learning_rate
        self.epochs = epochs
        self.realizations = realizations
        self.train_size = train_size
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.hit_rate = []
        self.tpr = [] # TPR: (Sensitivity, hit rate, recall)
        self.spc = [] # TNR=SPC: (Specificity)
        self.ppv = [] # PPV: Pos Pred Value (Precision)
        self.acc = 0 # Accuracy
        self.std = 0
        
    def initWeigths(self):
        # initializes weights randomly
        self.w = np.random.rand(self.output_layer, self.attributes+1)
    
    def degreeFunction(self, u):
        return np.array([1]) if u >= 0 else np.array([0])

    def activationFunction(self, u):
        try:
            if self.activation == 'degree':
                return self.degreeFunction(u)
        except:
            print('Error, activation function is not defined!')
        

    def predict(self, xi):
        u = np.dot(self.w, xi)
        y = self.activationFunction(u)
        return y

    def train(self):
        stop_error = 1
        cont_epochs = 0
        while (stop_error and cont_epochs < self.epochs):
            stop_error = 0
            self.x_train, self.y_train = shuffleData(self.x_train, self.y_train)
            (m, _) = self.x_train.shape
            for i in range(m):
                xi = self.x_train[i]
                y = self.predict(xi)

                d = self.y_train[i]
                error = d - y

                # check if this error
                if not np.array_equal(error, [0]):
                    stop_error = 1

                self.w += self.eta * (error * xi)

            cont_epochs += 1
        #print('Number of epochs: {}'.format(cont_epochs), end='\n')

    def test(self):
        (m, _) = self.x_test.shape
        y_actu = []
        y_pred = []
        for i in range(m):
            xi = self.x_test[i]
            y = self.predict(xi)

            d = self.y_test[i]
            error = d - y

            # Confusion Matrix
            y_actu.append(int(d))
            y_pred.append(int(y))

        cm = ConfusionMatrix(y_actu, y_pred)
        self.hit_rate.append(cm.ACC)
        self.tpr.append(cm.TPR)
        self.spc.append(cm.SPC)
        self.ppv.append(cm.PPV)
        #cm.print_stats()
        #print('Confusion Matrix:\n%s'.format(cm))

    def perceptron(self):
        self.x_data = normalizeData(self.x_data)
        self.x_data = insertBias(self.x_data)

        for i in range(self.realizations):
            self.initWeigths()
            x_data_aux, y_data_aux = shuffleData(self.x_data, self.y_data)
            self.x_train, self.x_test, self.y_train, self.y_test = splitData(x_data_aux, \
                y_data_aux, self.train_size)
            self.train()
            self.test()

        self.acc = np.mean(self.hit_rate)
        self.std = np.std(self.hit_rate)
        self.tpr = np.mean(self.tpr)
        self.spc = np.mean(self.spc)
        self.ppv = np.mean(self.ppv)

        print('Hit rate: {}'.format(self.hit_rate))
        print('Accuracy: {:.3f}'.format(self.acc*100))
        print('Standard deviation: {:.3f}'.format(self.std))
        print('Sensitivity: {:.3f}'.format(self.tpr*100))
        print('Specificity: {:.3f}'.format(self.spc*100))
        print('Precision: {:.3f}'.format(self.ppv*100))


