import random
import numpy as np

class Perceptron:

    def __init__(self, x_data, y_data, realizations=10, learning_rate=0.05, epochs=1, outLayer=1, trainSize=0.8):
        self.x_data = x_data
        self.y_data = y_data
        self.attributes = x_data.shape[1]
        self.eta = learning_rate
        self.epochs = epochs
        self.realizations = realizations
        self.outLayer = outLayer
        self.trainSize = trainSize
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.hit_rate = []
        self.accuracy = 0
        self.std = 0

    def bias(self):
        (m, _) = self.x_data.shape
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
        return np.array([1]) if u >= 0 else np.array([0])
        #if u >= 0:
        #    return np.array([1])
        #return np.array([0])
    
    def shuffle(self, x, y):
        data = list(zip(x, y))
        random.shuffle(data)
        x, y = zip(*data)
        return np.array(x), np.array(y)
    
    def split(self):
        self.x_data, self.y_data = self.shuffle(self.x_data, self.y_data)

        (m, _) = self.x_data.shape
        x = int(m * self.trainSize) 

        self.x_train = self.x_data[0:x]
        self.x_test = self.x_data[x:]
        self.y_train = self.y_data[0:x]
        self.y_test = self.y_data[x:]
    
    def predict(self, xi):
        u = np.dot(self.w, xi)
        y = self.degrau(u)
        return y

    def train(self):
        stop_error = 1
        cont_epochs = 0
        while (stop_error and cont_epochs < self.epochs):
            stop_error = 0
            self.x_train, self.y_train = self.shuffle(self.x_train, self.y_train)
            (m, _) = self.x_train.shape
            for i in range(m):
                xi = self.x_train[i]
                d = self.y_train[i]

                u = np.dot(self.w, xi)
                y = self.degrau(u)
                error = d - y

                # check if this error
                if not np.array_equal(error, [0]):
                    stop_error = 1

                self.w += self.eta * (error * xi)

            cont_epochs += 1

    def test(self):
        hits = 0.0
        (m, _) = self.x_test.shape
        for i in range(m):
            xi = self.x_test[i]
            y = self.predict(xi)

            d = self.y_test[i]
            error = d - y

            if np.array_equal(error, [0]):
                hits += 1
        
        self.hit_rate.append(hits/m)


    def perceptron(self):
        self.initWeigths()
        self.normalize()
        self.bias()

        for i in range(self.realizations):
            self.split()
            self.train()
            self.test()

        self.accuracy = np.mean(self.hit_rate)
        print('Accuracy {:.3f}'.format(self.accuracy*100))
        self.std = np.std(self.hit_rate)
        print('Std {:.3f}'.format(self.std))


