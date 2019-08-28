import random
import numpy as np

class Perceptron:

    def __init__(self, x_data, y_data, activation='linear', realizations=1, \
        learning_rate=0.05, epochs=200, outLayer=1, trainSize=0.8):
        self.x_data = x_data
        self.y_data = y_data
        self.activation = activation
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

    def insertBias(self):
        (m, _) = self.x_data.shape
        bias = -1 * np.ones((m,1))
        self.x_data = np.concatenate((bias, self.x_data), axis=1)
    
    def initWeigths(self):
        # initializes weights randomly
        self.w = np.random.rand(self.outLayer, self.attributes+1)
    
    def normalizeData(self):
        max_ = self.x_data.max(axis=0)
        min_ = self.x_data.min(axis=0)
        self.x_data = (self.x_data - min_) / (max_ - min_)
    
    def shuffleData(self, x, y):
        data = list(zip(x, y))
        random.shuffle(data)
        x, y = zip(*data)
        return np.array(x), np.array(y)
    
    def splitData(self):
        self.x_data, self.y_data = self.shuffleData(self.x_data, self.y_data)

        (m, _) = self.x_data.shape
        x = int(m * self.trainSize) 

        self.x_train = self.x_data[0:x]
        self.x_test = self.x_data[x:]
        self.y_train = self.y_data[0:x]
        self.y_test = self.y_data[x:]

    def degrau(self, u):
        return np.array([1]) if u >= 0 else np.array([0])

    def activationFunction(self, u):
        try:
            if self.activation == 'linear':
                return self.degrau(u)
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
            self.x_train, self.y_train = self.shuffleData(self.x_train, self.y_train)
            (m, _) = self.x_train.shape
            for i in range(m):
                xi = self.x_train[i]
                d = self.y_train[i]

                u = np.dot(self.w, xi)

                y = self.activationFunction(u)
                error = d - y

                # check if this error
                if not np.array_equal(error, [0]):
                    stop_error = 1

                self.w += self.eta * (error * xi)

            cont_epochs += 1
        #print('Number of epochs: {}'.format(cont_epochs), end='\n')

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
        self.normalizeData()
        self.insertBias()

        for i in range(self.realizations):
            self.initWeigths()
            self.splitData()
            self.train()
            self.test()

        print('Hit rate: {}'.format(self.hit_rate))
        self.accuracy = np.mean(self.hit_rate)
        self.std = np.std(self.hit_rate)

        print('Accuracy: {:.3f}'.format(self.accuracy*100))
        print('Standard deviation: {:.3f}'.format(self.std))


