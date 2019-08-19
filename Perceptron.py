import numpy as np
import random

class Perceptron:

    def __init__(self, x_data, y_data, realizations=1, learning_rate=0.05, epochs=1, outLayer=1):
        self.x_data = x_data
        self.y_data = y_data
        self.attributes = x_data.shape[1]
        self.eta = learning_rate
        self.epochs = epochs
        self.realizations = realizations
        self.outLayer = outLayer
        self.initWeigths()
        self.normalize()
        self.bias() # insert bias (-1)

    def bias(self):
        (m, n) = self.x_data.shape
        bias = -1 * np.ones((m,1))
        self.x_data = np.concatenate((bias, self.x_data), axis=1)
    
    def initWeigths(self):
        # initializes weights randomly
        self.w = np.random.rand(self.outLayer, self.attributes+1)
    
    def normalize(self):
        max_ = self.x_data.max(axis=0)
        min_ = self.x_data.min(axis=0)
        self.x_data = (self.x_data - min_) / (max_ - min_)

    def degrau(self, u):
        if u >= 0:
            return np.array([1])
        return np.array([0])
    
    def shuffle(self, a, b):
        c = list(zip(a, b))
        random.shuffle(c)
        a, b = zip(*c)
        return np.array(a), np.array(b)

    def train(self):
        stop_error = 1
        cont_epochs = 0
        while (stop_error and cont_epochs < self.epochs):

            #self.x_data, self.y_data = self.shuffle(self.x_data, self.y_data)
            stop_error = 0
            (m, n) = self.x_data.shape
            for i in range(m):
                xi = self.x_data[i]
                d = self.y_data[i]

                u = np.dot(self.w, xi)
                y = self.degrau(u)
                error = d - y

                #if error != 0:
                if not np.array_equal(error, [0]):
                    stop_error = 1

                self.w += self.eta * (error * xi)

            cont_epochs += 1
# end Perceptron




#x_train, x_test, y_train, y_test = split(x_data, y_data, test_size=0.2)