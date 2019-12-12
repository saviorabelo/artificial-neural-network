import math
import random
import numpy as np
from Utils.utils import Util as util
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score as acc
#TPR: (Sensitivity, hit rate, recall)
from sklearn.metrics import recall_score as tpr
#TNR=SPC: (Specificity)
#PPV: Pos Pred Value (Precision)
from sklearn.metrics import precision_score as ppv

class RBF:
    def __init__(self, x_data, y_data, g_search=False, n_centers=10, width=10):
        self.x_data = x_data
        self.y_data = y_data
        self.n_classes = np.unique(self.y_data, axis=0)
        self.g_search = g_search
        self.attributes = x_data.shape[1]
        self.output_layer = y_data.shape[1]
        self.n_centers = n_centers
        self.width = width
        self.realizations = 1
        self.train_size = 0.8
        self.mse = []
        self.rmse = []
    
    def initWeigths(self, n_centers):
        params = {}
        a = 0
        b = 1
        params['c'] = (b - a) * np.random.random_sample((n_centers, self.attributes+1)) + a
        return params

    def predict(self, xi, params, n_centers, width):

        c = params['c']
        w = params['w']
        
        h = np.zeros((1, n_centers))

        for j in range(n_centers):
            h[0,j] = self.saidas_centro(xi, c[j], width)

        h = np.concatenate(([-1], h), axis=None)
        y = np.dot(h, w)
        
        return y

    def saidas_centro(self, x, c, width):
        aux = (x - c).reshape(-1,1).T
        ans = np.exp(-0.5 * np.dot(aux, aux.T) / (width^2) )
        return ans

    def train(self, x_train, y_train, n_centers, width):

        params = self.initWeigths(n_centers)
        c = params['c']
        
        x_train, y_train = util.shuffleData(x_train, y_train)
        (p, _) = x_train.shape
        h = np.zeros((p, n_centers))

        for i in range(p):
            for j in range(n_centers):
                h[i,j] = self.saidas_centro(x_train[i], c[j], width)

        bias = -1 * np.ones((p, 1))
        h = np.concatenate((bias, h), axis=1)
        w = np.dot(np.linalg.pinv(h), y_train)

        params['w'] = w
        return params

    def test(self, x_test, y_test, params, n_centers, width):
        error_epoch = 0.0
        (p, _) = x_test.shape
        for k in range(p):
            x_k = x_test[k]
            y = self.predict(x_test[k], params, n_centers, width)
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
            
            best_n_centers = self.n_centers
            best_width = self.width
            params = self.train(x_train, y_train, best_n_centers, best_width)
            mse, rmse = self.test(x_test, y_test, params, best_n_centers, best_width)

            self.mse.append(mse)
            self.rmse.append(rmse)

        print('{} Realizations'.format(self.realizations))
        print('MSE: {}'.format(np.mean(self.mse)))
        print('Std MSE: {}'.format(np.std(self.mse)))
        print('RMSE: {}'.format(np.mean(self.rmse)))
        print('Std RMSE: {}'.format(np.std(self.rmse)))

        #self.plotRegression(x_data, x_train, x_test, y_train, y_test, self.predict, params, best_n_centers, best_width)

    def plotRegression(self, x_data, x_train, x_test, y_train, y_test, predict, params, n_centers, width):
        x = x_data[:,1]
        y = [predict(np.array([-1, i]), params, n_centers, width) for i in x]

        fig, ax = plt.subplots()
        plt.title('RBF Color Map')
        plt.xlabel('Eixo X')
        plt.ylabel('Eixo y')

        ax.scatter(x, y, label='Predict', color=[0.31, 0.31, 0.31])
        ax.scatter(x_train[:,1], y_train[:,0], label='Train Data', color=[0.00, 0.45, 0.74])
        ax.scatter(x_test[:,1], y_test[:,0], label='Test Data', color='green')

        ax.legend()
        ax.grid(True)
        plt.show()

        #fig.savefig('.\RadialBasisFunctionRegression\Results\sin_1.png')


