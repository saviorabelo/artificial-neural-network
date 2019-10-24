import math
import random
import numpy as np
from Utils.utils import Util as util
from matplotlib import pyplot as plt

class Perceptron:
    def __init__(self, x_data, y_data, activation, hidden_layer):
        self.x_data = x_data
        self.y_data = y_data
        self.n_classes = np.unique(self.y_data, axis=0)
        self.attributes = x_data.shape[1]
        self.hidden_layer = hidden_layer
        self.output_layer = y_data.shape[1]
        self.epochs = 500
        self.realizations = 1
        self.precision = 10**(-7)
        self.train_size = 0.8
        self.activation = activation
        self.mse = []
        self.rmse = []
    
    def initWeigths(self, hidden_layer):
        params = {}
        a = -1.5
        b = 1.5
        params['w'] = (b - a) * np.random.random_sample((self.attributes+1, hidden_layer)) + a
        params['m'] = (b - a) * np.random.random_sample((hidden_layer+1, self.output_layer)) + a
        return params

    def updateEta(self, epoch):
        eta_i = 0.1
        eta_f = 0.05
        eta = eta_i * ((eta_f / eta_i) ** (epoch / self.epochs))
        self.eta = eta

    def function(self, u):
        if self.activation == 'logistic':
            y = 1.0/(1.0 + np.exp(-u))
        elif self.activation == 'tanh':
            y = (np.exp(u) - np.exp(-u))/(np.exp(u) + np.exp(-u))
        else:
            raise ValueError('Error in function!')
            y = 0
        return y    

    def derivate(self, u):
        if self.activation == 'logistic':
            y_ = u * (1.0 - u)
        elif self.activation == 'tanh':
            y_ = 0.5 * (1.0 - (u * u))
        else:
            raise ValueError('Error in derivate!')
            y_ = 0
        return y_

    def predict(self, xi, params):
        w = params['w']
        m = params['m']
        
        H = np.dot(xi, w)
        H = self.function(H)
        H = np.concatenate(([-1], H), axis=None)
        Y = np.dot(H, m)

        return Y

    def train(self, x_train, y_train, hidden_layer):
        error_old = 0
        cont_epochs = 0
        mse_vector = []
        params = self.initWeigths(hidden_layer)
        w = params['w']
        m = params['m']
        while True:
            self.updateEta(cont_epochs)
            x_train, y_train = util.shuffleData(x_train, y_train)
            (p, _) = x_train.shape
            error_epoch = 0
            for k in range(p):
                x_k = x_train[k]
                H = np.dot(x_k, w)
                H = self.function(H)
                H_ = self.derivate(H)
                
                H = np.concatenate(([-1], H), axis=None)
                Y = np.dot(H, m)

                # Quadratic Error Calculation
                d = y_train[k]
                error = d - Y
                error_epoch += np.sum(error**2)
                
                # Output layer
                delta_output = (error).reshape(-1, 1)
                aux_output = (self.eta * delta_output)
                m += np.dot(H.reshape(-1, 1), aux_output.T)

                # Hidden layer
                delta_hidden = np.sum(np.dot(m, delta_output)) * H_
                aux_hidden = (self.eta * delta_hidden).reshape(-1, 1)
                w += np.dot(x_k.reshape(-1, 1), aux_hidden.T)

            mse = (0.5 * error_epoch) / p
            mse_vector.append(mse)

            if abs(error_epoch - error_old) <= self.precision:
                #print('Stop Precision: {} (Epochs {})'.format(abs(error_epoch - error_old), cont_epochs))
                break
            if cont_epochs >= self.epochs:
                #print('Stop Epochs: {}'.format(cont_epochs))
                break
        
            error_old = error_epoch
            cont_epochs += 1
        #util.plotErrors(mse_vector)
        params['w'] = w
        params['m'] = m
        return params

    def test(self, x_test, y_test, params):
        error_epoch = 0.0
        (p, _) = x_test.shape
        for k in range(p):
            x_k = x_test[k]
            y = self.predict(x_k, params)
            d = y_test[k]
           
            error = d - y
            error_epoch += error**2
        
        mse = error_epoch / p
        rmse = np.sqrt(mse)
        return mse, rmse

    def execute(self):
        x_data = util.insertBias(self.x_data)
        y_data = self.y_data

        for i in range(self.realizations):
            x_data_aux, y_data_aux = util.shuffleData(x_data, y_data)
            x_train, x_test, y_train, y_test = util.splitData(x_data_aux, y_data_aux, self.train_size)
            
            best_hidden_layer = self.hidden_layer
            params = self.train(x_train, y_train, best_hidden_layer)
            mse, rmse = self.test(x_test, y_test, params)

            self.mse.append(mse)
            self.rmse.append(rmse)

        print('{} Realizations'.format(self.realizations))
        print('MSE: {}'.format(np.mean(self.mse)))
        print('Std MSE: {}'.format(np.std(self.mse)))
        print('RMSE: {}'.format(np.mean(self.rmse)))
        print('Std RMSE: {}'.format(np.std(self.rmse)))

        #self.plotRegression(x_data, x_train, x_test, y_train, y_test, self.predict, params)

    def plotRegression(self, x_data, x_train, x_test, y_train, y_test, predict, params):
        x = x_data[:,1]
        y = [predict(np.array([-1, i]), params) for i in x]

        fig, ax = plt.subplots()
        plt.title('MLP Color Map')
        plt.xlabel('Eixo X')
        plt.ylabel('Eixo y')

        ax.scatter(x, y, label='Predict', color=[0.31, 0.31, 0.31])
        ax.scatter(x_train[:,1], y_train[:,0], label='Train Data', color=[0.00, 0.45, 0.74])
        ax.scatter(x_test[:,1], y_test[:,0], label='Test Data', color='green')

        ax.legend()
        ax.grid(True)
        plt.show()

        #fig.savefig('.\MultilayerPerceptronRegression\Results\sin_q10.png')
