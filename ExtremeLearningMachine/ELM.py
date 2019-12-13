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

class ELM:
    def __init__(self, x_data, y_data, activation='logistic', g_search=False, hidden_layer=10):
        self.x_data = x_data
        self.y_data = y_data
        self.n_classes = np.unique(self.y_data, axis=0)
        self.g_search = g_search
        self.attributes = x_data.shape[1]
        self.hidden_layer = hidden_layer
        self.output_layer = y_data.shape[1]
        self.epochs = 500
        self.realizations = 1
        self.precision = 10**(-5)
        self.train_size = 0.8
        self.activation = activation
        self.hit_rate = []
        self.tpr = []
        self.spc = []
        self.ppv = []
    
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

    def activationFunction(self, u):
        value = np.amax(u)
        y = np.where(u == value, 1, 0)
        return y

    def predict(self, xi, params):
        w = params['w']
        m = params['m']
        
        H = np.dot(xi, w)
        H = self.function(H)
        H = np.concatenate(([-1], H), axis=None)
        H = H.reshape(1,-1)

        Y = np.dot(H, m)
        Y = self.function(Y)
        y = self.activationFunction(Y)

        return y[0]

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
        y_true = []
        y_pred = []
        (p, _) = x_test.shape
        for k in range(p):
            x_k = x_test[k]
            y = self.predict(x_k, params)
            d = y_test[k]
           
            # Confusion Matrix
            y_true.append(list(d))
            y_pred.append(list(y))

        a = util.inverse_transform(y_true, self.n_classes)
        b = util.inverse_transform(y_pred, self.n_classes)
        #return acc(a,b), tpr(a,b, average='macro'), 0, ppv(a,b, average='weighted')
        return acc(a,b), 0, 0, 0

    def grid_search(self, x_train, y_train):
        (n, _) = x_train.shape
        hidden_layer = [2,4,6,8,10,12]
        k_fold = 5
        slice_ = int(n/k_fold)

        grid_accuracy = []
        for q in hidden_layer:
            scores = []
            # cross validation
            for j in range(k_fold):
                # set range
                a = j*slice_
                b = (j+1)*slice_

                X_tra_aux = np.concatenate((x_train[0:a], x_train[b:n]), axis=0)
                X_test_aux = x_train[a:b]
                Y_tra_aux = np.concatenate((y_train[0:a], y_train[b:n]), axis=0)
                Y_test_aux = y_train[a:b]

                params = self.train(X_tra_aux, Y_tra_aux, q)
                acc, _, _, _ = self.test(X_test_aux, Y_test_aux, params)
                scores.append(acc)
            grid_accuracy.append(np.mean(scores))
        print('Grid search:', grid_accuracy)
        index_max = np.argmax(grid_accuracy)
        return hidden_layer[index_max]

    def execute(self):
        x_data = util.normalizeData(self.x_data)
        x_data = util.insertBias(x_data)
        y_data = self.y_data

        for i in range(self.realizations):
            x_data_aux, y_data_aux = util.shuffleData(x_data, y_data)
            x_train, x_test, y_train, y_test = util.splitData(x_data_aux, y_data_aux, self.train_size)
            
            if self.g_search:
                best_hidden_layer = self.grid_search(x_train, y_train)
                print('Hidden Layer:', best_hidden_layer)
            else:
                best_hidden_layer = self.hidden_layer

            params = self.train(x_train, y_train, best_hidden_layer)
            acc, tpr, spc, ppv = self.test(x_test, y_test, params)
            
            self.hit_rate.append(acc)
            self.tpr.append(tpr)
            self.spc.append(spc)
            self.ppv.append(ppv)

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

        #self.plotColorMap_3C(x_train, x_test, y_train, self.predict, params)
        #self.plotColorMap_2C(x_train, x_test, y_train, self.predict, params)

    def plotColorMap_3C(self, x_train, x_test, y_train, predict, params):
        color1_x = []
        color1_y = []
        color2_x = []
        color2_y = []
        color3_x = []
        color3_y = []
        for i in np.arange(0,1.0,0.005):
            for j in np.arange(0,1.0,0.005):
                xi = np.array([-1, i, j])
                y = predict(xi, params)
                if np.array_equal(y, [0,0,1]):
                    color1_x.append(i)
                    color1_y.append(j)
                elif np.array_equal(y, [0,1,0]):
                    color2_x.append(i)
                    color2_y.append(j)
                elif np.array_equal(y, [1,0,0]):
                    color3_x.append(i)
                    color3_y.append(j)
                else:
                    raise ValueError('Error color!\n')
        
        # Split a train class
        i = []
        j = []
        k = []
        for index,y in enumerate(y_train):
            if np.array_equal(y, [0,0,1]):
                i.append(index)
            elif np.array_equal(y, [0,1,0]):
                j.append(index)
            elif np.array_equal(y, [1,0,0]):
                k.append(index)
            else:
                raise ValueError('Error!\n')
        train1 = x_train[i]
        train2 = x_train[j]
        train3 = x_train[k]

        fig, ax = plt.subplots()
        plt.title('ELM Color Map')
        plt.xlabel('Eixo X')
        plt.ylabel('Eixo y')

        ax.scatter(color1_x, color1_y, color=[0.80, 0.88, 0.97])
        ax.scatter(color2_x, color2_y, color=[0.80, 0.80, 0.80])
        ax.scatter(color3_x, color3_y, color=[0.95, 0.87, 0.73])
        ax.scatter(train1[:,1], train1[:,2], label='Classe 1', color=[0.00, 0.45, 0.74])
        ax.scatter(train2[:,1], train2[:,2], label='Classe 2', color=[0.31, 0.31, 0.31])
        ax.scatter(train3[:,1], train3[:,2], label='Classe 3', color=[0.60, 0.20, 0.00])
        ax.scatter(x_test[:,1], x_test[:,2], label='Test Data', color='green')

        ax.legend()
        ax.grid(True)
        plt.show()

        #fig.savefig('.\MultilayerPerceptron\Results\color_map.png')
    
    def plotColorMap_2C(self, x_train, x_test, y_train, predict, params):
        color1_x = []
        color1_y = []
        color2_x = []
        color2_y = []
        for i in np.arange(0,1,0.005):
            for j in np.arange(0,1,0.005):
                xi = np.array([-1, i, j])
                y = predict(xi, params)
                if np.array_equal(y, [0,1]):
                    color1_x.append(i)
                    color1_y.append(j)
                elif np.array_equal(y, [1,0]):
                    color2_x.append(i)
                    color2_y.append(j)
                else:
                    raise ValueError('Error color!\n')
        
        # Split a train class
        i = []
        j = []
        for index,y in enumerate(y_train):
            if np.array_equal(y, [0,1]):
                i.append(index)
            elif np.array_equal(y, [1,0]):
                j.append(index)
            else:
                raise ValueError('Error!\n')
        train1 = x_train[i]
        train2 = x_train[j]

        fig, ax = plt.subplots()
        plt.title('ELM Color Map')
        plt.xlabel('Eixo X')
        plt.ylabel('Eixo y')

        ax.scatter(color1_x, color1_y, color=[0.80, 0.88, 0.97])
        ax.scatter(color2_x, color2_y, color=[0.80, 0.80, 0.80])
        ax.scatter(train1[:,1], train1[:,2], label='Classe 1', color=[0.00, 0.45, 0.74])
        ax.scatter(train2[:,1], train2[:,2], label='Classe 2', color=[0.31, 0.31, 0.31])
        ax.scatter(x_test[:,1], x_test[:,2], label='Test Data', color='green')

        ax.legend()
        ax.grid(True)
        plt.show()

        #fig.savefig('.\MultilayerPerceptron\Results\color_map.png')
        #fig.savefig('.\MultilayerPerceptron\Results\color_map.png', dpi=fig.dpi, bbox_inches='tight')
