import numpy as np
import random

class Perceptron:

    def __init__(self, x_data, y_data, realizations=1, rate=0.05, epochs=1, outLayer=1):
        self.x_data = x_data
        self.y_data = y_data
        self.learning_rate = rate
        self.epochs = epochs
        self.realizations = realizations
        self.attributes = x_data.shape[1]
        self.outLayer = outLayer
        self.w = self.initWeigths()
    
    def initWeigths(self):
        # initializes weights randomly
        self.w = np.random.random((self.attributes, self.outLayer))
    
    def normalize(self):
        max_ = self.x_data.max(axis=0)
        min_ = self.x_data.min(axis=0)
        self.x_data = (self.x_data - min_) / (max_ - min_)

    def train(self):
        stop_error = 0
        cont_epochs = 0
        while (stop_error or cont_epochs < self.epochs):
            for xi in (self.x_data):
                
                u = xi * self.w
                y = degrau(u)
                error = d - y

                if error != 0:
                    stop_error = 1

            cont_epochs += 1




#x_train, x_test, y_train, y_test = split(x_data, y_data, test_size=0.2)