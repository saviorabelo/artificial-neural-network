import random
import numpy as np
from Utils.utils import *
from matplotlib import pyplot as plt
from pandas_ml import ConfusionMatrix

class Perceptron:

    def __init__(self, x_data, y_data, normalize=None):
        self.x_data = x_data
        self.y_data = y_data
        self.activation = 'degree'
        self.attributes = x_data.shape[1]
        self.output_layer = y_data.shape[1]
        self.eta = 0.0 # Learning rate
        self.epochs = 200
        self.realizations = 1
        self.train_size = 0.8
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.hit_rate = []
        self.tpr = [] # TPR: (Sensitivity, hit rate, recall)
        self.spc = [] # TNR=SPC: (Specificity)
        self.ppv = [] # PPV: Pos Pred Value (Precision)
        self.acc = 0 # Accuracy
        self.std = 0
        if normalize == None:
            self.normalize = True
        else:
            self.normalize = normalize
    
    def initWeigths(self):
        # initializes weights randomly
        self.w = np.random.rand(self.output_layer, self.attributes+1)
    
    def degreeFunction(self, u):
        return np.array([1]) if u >= 0 else np.array([0])

    def activationFunction(self, u):
        try:
            if self.activation == 'degree':
                return self.degreeFunction(u)
        except:
            print('Error, activation function is not defined!')

    def predict(self, xi):
        u = np.dot(self.w, xi)
        y = self.activationFunction(u)
        return y
    
    def updateEta(self, epoch):
        eta_i = 0.05
        eta_f = 0.5
        eta = eta_f * ((eta_i / eta_f) ** (epoch / self.epochs))
        self.eta = eta

    def train(self):
        stop_error = 1
        cont_epochs = 0
        vector_error = []
        while (stop_error and cont_epochs < self.epochs):
            self.updateEta(cont_epochs)
            stop_error = 0
            self.x_train, self.y_train = shuffleData(self.x_train, self.y_train)
            (m, _) = self.x_train.shape
            aux = 0
            for i in range(m):
                xi = self.x_train[i]
                y = self.predict(xi)

                d = self.y_train[i]
                error = d - y
                aux += abs(int(error))

                # check if this error
                if not np.array_equal(error, [0]):
                    stop_error = 1

                # update weights
                self.w += self.eta * (error * xi)
            vector_error.append(aux)
            cont_epochs += 1
        #plotErrors(vector_error)

    def test(self):
        (m, _) = self.x_test.shape
        y_actu = []
        y_pred = []
        for i in range(m):
            xi = self.x_test[i]
            y = self.predict(xi)

            d = self.y_test[i]
            #error = d - y

            # Confusion Matrix
            y_actu.append(int(d))
            y_pred.append(int(y))

        cm = ConfusionMatrix(y_actu, y_pred)
        self.hit_rate.append(cm.ACC)
        self.tpr.append(cm.TPR)
        self.spc.append(cm.SPC)
        self.ppv.append(cm.PPV)
        #cm.print_stats()
        #plotConfusionMatrix(cm)
 
    def perceptron(self):
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

    def plotColorMap(self):

        if self.attributes == 2:
            color1_x = []
            color1_y = []
            color2_x = []
            color2_y = []
            for i in np.arange(0,1.0,0.005):
                for j in np.arange(0,1.0,0.005):
                    xi = np.array([-1, i, j])
                    y = self.predict(xi)
                    if np.array_equal(y, [0]):
                        color1_x.append(i)
                        color1_y.append(j)
                    else:
                        color2_x.append(i)
                        color2_y.append(j)
            
            # Split a train class
            i, _ = np.where(self.y_train == [0])
            train1 = self.x_train[i]
            j, _ = np.where(self.y_train == [1])
            train2 = self.x_train[j]

            fig, ax = plt.subplots()
            plt.title('Perceptron Color Map')
            plt.xlabel('Eixo X')
            plt.ylabel('Eixo y')

            ax.scatter(color1_x, color1_y, color=[0.80, 0.88, 0.97])
            ax.scatter(color2_x, color2_y, color=[0.80, 0.80, 0.80])
            ax.scatter(train1[:,1], train1[:,2], label='Classe 1', color=[0.00, 0.45, 0.74])
            ax.scatter(train2[:,1], train2[:,2], label='Classe 2', color=[0.31, 0.31, 0.31])
            ax.scatter(self.x_test[:,1], self.x_test[:,2], label='Test Data', color='green')

            ax.legend()
            ax.grid(True)
            plt.show()
        else:
            print('Invalid number of attributes!\n')
