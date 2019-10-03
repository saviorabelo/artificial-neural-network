import random
import numpy as np
from Utils.utils import Util as util
from matplotlib import pyplot as plt
from pandas_ml import ConfusionMatrix

class Perceptron:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.activation = 'degree'
        self.attributes = x_data.shape[1]
        self.output_layer = y_data.shape[1]
        self.epochs = 200
        self.realizations = 1
        self.train_size = 0.8
        self.hit_rate = []
        self.tpr = []
        self.spc = []
        self.ppv = []
    
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

    def train(self, x_train, y_train):
        stop_error = 1
        cont_epochs = 0
        vector_error = []
        while (stop_error and cont_epochs < self.epochs):
            self.updateEta(cont_epochs)
            stop_error = 0
            x_train, y_train = util.shuffleData(x_train, y_train)
            (m, _) = x_train.shape
            aux = 0
            for i in range(m):
                xi = x_train[i]
                y = self.predict(xi)

                d = y_train[i]
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

    def test(self, x_test, y_test):
        y_actu = []
        y_pred = []
        (m, _) = x_test.shape
        for i in range(m):
            xi = x_test[i]
            y = self.predict(xi)
            d = y_test[i]
            #error = d - y

            # Confusion Matrix
            y_actu.append(int(d))
            y_pred.append(int(y))

        cm = ConfusionMatrix(y_actu, y_pred)
        #cm.print_stats()
        #plotConfusionMatrix(cm)

        return cm.ACC, cm.TPR, cm.SPC, cm.PPV 
 
    def perceptron(self):
        # Pre processing
        x_data = util.normalizeData(self.x_data)
        x_data = util.insertBias(x_data)
        y_data = self.y_data

        for i in range(self.realizations):
            self.initWeigths()
            x_data_aux, y_data_aux = util.shuffleData(x_data, y_data)
            x_train, x_test, y_train, y_test = util.splitData(x_data_aux, y_data_aux, self.train_size)
            self.train(x_train, y_train)
            acc, tpr, spc, ppv = self.test(x_test, y_test)
            
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
