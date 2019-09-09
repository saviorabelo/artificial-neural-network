import random
import numpy as np
from Utils.utils import *
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Adaline:

    def __init__(self, x_data, y_data, normalize=None):
        self.x_data = x_data
        self.y_data = y_data
        self.activation = 'linear'
        self.attributes = x_data.shape[1]
        self.output_layer = y_data.shape[1]
        self.eta = 0.0
        self.epochs = 200
        self.realizations = 1
        self.precision = 10**(-7)
        self.train_size = 0.8
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.mse = 0.0
        self.rmse = 0.0
        if normalize == None:
            self.normalize = True
        else:
            self.normalize = normalize
    
    def initWeigths(self):
        # initializes weights randomly
        self.w = np.random.rand(self.output_layer, self.attributes+1)

    def activationFunction(self, u):
        try:
            if self.activation == 'linear':
                return u
        except:
            print('Error, activation function is not defined!')
    
    def updateEta(self, epoch):
        eta_i = 0.05
        eta_f = 0.5
        self.eta = eta_f * ((eta_i / eta_f) ** (epoch / self.epochs))
    
    def predict(self, xi):
        u = np.dot(self.w, xi)
        y = self.activationFunction(u)
        return y

    def train(self):
        error_old = 0
        cont_epochs = 0
        mse_vector = []
        while True:
            self.updateEta(cont_epochs)
            self.x_train, self.y_train = shuffleData(self.x_train, self.y_train)
            (m, _) = self.x_train.shape
            error_epoch = 0
            for i in range(m):
                xi = self.x_train[i]
                y = self.predict(xi)
                d = self.y_train[i]

                error = d - y
                error_epoch += error**2

                # update weights
                self.w += self.eta * (error * xi)

            mse = error_epoch / m
            mse_vector.append(mse)

            if abs(error_epoch - error_old) <= self.precision:
                print('Stop Precision: {}'.format(abs(error_epoch - error_old)))
                break
            if cont_epochs >= self.epochs:
                print('Stop Epochs: {}'.format(cont_epochs))
                break

            error_old = error_epoch
            cont_epochs += 1
        #plotErrors(mse_vector)

    def test(self):
        (m, _) = self.x_test.shape
        error_epoch = 0.0
        for i in range(m):
            xi = self.x_test[i]
            y = self.predict(xi)

            d = self.y_test[i]
            error = d - y
            error_epoch += error**2

        self.mse = error_epoch / m
        self.rmse = np.sqrt(self.mse)

    def adaline(self):
        if self.normalize:
            self.x_data = normalizeData(self.x_data)
        self.x_data = insertBias(self.x_data)

        for i in range(self.realizations):
            self.initWeigths()
            x_data_aux, y_data_aux = shuffleData(self.x_data, self.y_data)
            self.x_train, self.x_test, self.y_train, self.y_test = splitData(x_data_aux, \
                y_data_aux, self.train_size)
            self.train()
            self.test()

        print('MSE: {}'.format(self.mse))
        print('RMSE: {}'.format(self.rmse))
    
    def plotColorMap(self):
        if self.attributes == 1:
            self.plotColorMap2D()
        elif self.attributes == 2:
            self.plotColorMap3D()
        else:
            print('Invalid number of attributes!\n')
    
    def plotColorMap2D(self):
        x = self.x_data[:,1]
        y = [self.predict(np.array([-1, i])) for i in x]

        fig, ax = plt.subplots()
        plt.title('Adaline 2D Color Map\nMSE: {}\nRMSE: {}'.format(self.mse, self.rmse))
        plt.xlabel('Eixo X')
        plt.ylabel('Eixo y')

        ax.scatter(x, y, label='Predict', color=[0.31, 0.31, 0.31])
        ax.scatter(self.x_train[:,1], self.y_train[:,0], label='Train Data', color=[0.00, 0.45, 0.74])
        ax.scatter(self.x_test[:,1], self.y_test[:,0], label='Test Data', color='green')

        ax.legend()
        ax.grid(True)
        plt.show()

    def plotColorMap3D(self):
        x = self.x_data[:,1]
        y = self.x_data[:,2]
        z = []
        for i in range(len(x)):
            aux = self.predict([-1, x[i], y[i]])
            z.append(aux[0])
        z = np.array(z)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, cmap=plt.cm.PuOr)
        #ax.scatter(x, y, z, label='Predict', c='k', marker='o')
        ax.scatter(self.x_train[:,1], self.x_train[:,2], self.y_train[:,0], label='Train Data', color=[0.00, 0.45, 0.74])
        ax.scatter(self.x_test[:,1], self.x_test[:,2], self.y_test[:,0], label='Test Data', color='green')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.title('Adaline 3D Color Map\nMSE: {}\nRMSE: {}'.format(self.mse, self.rmse))
        plt.show()