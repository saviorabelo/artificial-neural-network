import math
import random
import numpy as np
from Utils.utils import Util as util
from matplotlib import pyplot as plt


class ELM:
    def __init__(self, x_data, y_data, activation='logistic', g_search=False, hidden_layer=10):
        self.x_data = x_data
        self.y_data = y_data
        self.n_classes = np.unique(self.y_data, axis=0)
        self.g_search = g_search
        self.attributes = x_data.shape[1]
        self.hidden_layer = hidden_layer
        self.output_layer = y_data.shape[1]
        self.realizations = 1
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

    def function(self, u):
        if self.activation == 'logistic':
            y = 1.0/(1.0 + np.exp(-u))
        elif self.activation == 'tanh':
            y = (np.exp(u) - np.exp(-u))/(np.exp(u) + np.exp(-u))
        else:
            raise ValueError('Error in function!')
            y = 0
        return y    

    def predict(self, xi, params):
        w = params['w']
        m = params['m']
        
        H = np.dot(xi, w)
        H = self.function(H)
        H = np.concatenate(([-1], H), axis=None)
        H = H.reshape(1,-1)

        Y = np.dot(H, m)

        return Y

    def train(self, x_train, y_train, hidden_layer):
        error_old = 0
        cont_epochs = 0
        mse_vector = []
        params = self.initWeigths(hidden_layer)
        w = params['w']
        m = params['m']

        H = np.dot(x_train, w)
        H = self.function(H)

        # Bias
        (m, _) = H.shape
        bias = -1 * np.ones((m, 1))
        H = np.concatenate((bias, H), axis=1)

        H_pinv = np.linalg.pinv(H)
        m = np.dot(H_pinv, y_train)

        params['w'] = w
        params['m'] = m
        return params

    def test(self, x_test, y_test, params):
        error_epoch = 0.0
        (p, _) = x_test.shape
        for k in range(p):
            x_k = x_test[k]
            y = self.predict(x_test[k], params)
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
        plt.title('ELM Color Map')
        plt.xlabel('Eixo X')
        plt.ylabel('Eixo y')

        ax.scatter(x, y, label='Predict', color=[0.31, 0.31, 0.31])
        ax.scatter(x_train[:,1], y_train[:,0], label='Train Data', color=[0.00, 0.45, 0.74])
        ax.scatter(x_test[:,1], y_test[:,0], label='Test Data', color='green')

        ax.legend()
        ax.grid(True)
        plt.show()

        #fig.savefig('.\RadialBasisFunctionRegression\Results\sin_1.png')


