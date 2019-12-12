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
        self.hit_rate = []
        self.tpr = []
        self.spc = []
        self.ppv = []
    
    def initWeigths(self, n_centers):
        params = {}
        a = 0
        b = 1
        params['c'] = (b - a) * np.random.random_sample((n_centers, self.attributes+1)) + a
        return params

    def activationFunction(self, u):
        value = np.amax(u)
        y = np.where(u == value, 1, 0)
        return y

    def predict(self, xi, params, n_centers, width):

        c = params['c']
        w = params['w']
        
        h = np.zeros((1, n_centers))

        for j in range(n_centers):
            h[0,j] = self.saidas_centro(xi, c[j], width)

        h = np.concatenate(([-1], h), axis=None)
        Y = np.dot(h, w)
        y = self.activationFunction(Y)
        
        return y

    def grid_search(self, x_train, y_train):
        (n, _) = x_train.shape
        center = [10,12,14,16,18,20]
        width = [10,12,14,16,18,20]
        k_fold = 5
        slice_ = int(n/k_fold)

        grid_accuracy = np.zeros((len(width), len(center)))
        for index_w, w in enumerate(width):
            for index_c, c in enumerate(center):
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

                    params = self.train(X_tra_aux, Y_tra_aux, c, w)
                    acc, _, _, _ = self.test(X_test_aux, Y_test_aux, params, c, w)
                    scores.append(acc)
                grid_accuracy[index_w, index_c] = np.mean(scores)
        #print('Grid search:', grid_accuracy)
        ind = np.unravel_index(np.argmax(grid_accuracy, axis=None), grid_accuracy.shape)
        return width[ind[0]], center[ind[1]]
    
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
        y_true = []
        y_pred = []
        (p, _) = x_test.shape
        for i in range(p):
            d = y_test[i]
            y = self.predict(x_test[i], params, n_centers, width)
            
            # Confusion Matrix
            y_true.append(list(d))
            y_pred.append(list(y))

        a = util.inverse_transform(y_true, self.n_classes)
        b = util.inverse_transform(y_pred, self.n_classes)
        #return acc(a,b), tpr(a,b, average='macro'), 0, ppv(a,b, average='weighted')
        return acc(a,b), 0, 0, 0

    def execute(self):
        x_data = util.normalizeData(self.x_data)
        x_data = util.insertBias(x_data)
        y_data = self.y_data

        for i in range(self.realizations):
            x_data_aux, y_data_aux = util.shuffleData(x_data, y_data)
            x_train, x_test, y_train, y_test = util.splitData(x_data_aux, y_data_aux, self.train_size)
            
            if self.g_search:
                best_n_centers, best_width = self.grid_search(x_train, y_train)
                print('Best N Centers: ', best_n_centers)
                print('Best Width: ', best_width)
            else:
                best_n_centers = self.n_centers
                best_width = self.width

            params = self.train(x_train, y_train, best_n_centers, best_width)
            acc, tpr, spc, ppv = self.test(x_test, y_test, params, best_n_centers, best_width)
            
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

        #self.plotColorMap_3C(x_train, x_test, y_train, self.predict, params, best_n_centers, best_width)
        #self.plotColorMap_2C(x_train, x_test, y_train, self.predict, params, best_n_centers, best_width)

    def plotColorMap_3C(self, x_train, x_test, y_train, predict, params, best_n_centers, best_width):
        color1_x = []
        color1_y = []
        color2_x = []
        color2_y = []
        color3_x = []
        color3_y = []
        for i in np.arange(0,1.0,0.005):
            for j in np.arange(0,1.0,0.005):
                xi = np.array([-1, i, j])
                y = predict(xi, params, best_n_centers, best_width)
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
        plt.title('MLP Color Map')
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
    
    def plotColorMap_2C(self, x_train, x_test, y_train, predict, params, best_n_centers, best_width):
        color1_x = []
        color1_y = []
        color2_x = []
        color2_y = []
        for i in np.arange(0,1,0.005):
            for j in np.arange(0,1,0.005):
                xi = np.array([-1, i, j])
                y = predict(xi, params, best_n_centers, best_width)
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
        plt.title('MLP Color Map')
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
