import random
import numpy as np
from matplotlib import pyplot as plt

class Util:
    def insertBias(data):
        (m, _) = data.shape
        bias = -1 * np.ones((m,1))
        data = np.concatenate((bias, data), axis=1)
        return data
        
    def normalizeData(data):
        max_ = data.max(axis=0)
        min_ = data.min(axis=0)
        data = (data - min_) / (max_ - min_)
        return data

    def shuffleData(x, y):
        data = list(zip(x, y))
        random.shuffle(data)
        x_aux, y_aux = zip(*data)
        x_aux, y_aux = np.array(x_aux), np.array(y_aux)
        return x_aux, y_aux

    def splitData(x_data, y_data, train_size):
        (m, _) = x_data.shape
        x = int(m * train_size) 

        x_train = x_data[0:x]
        x_test = x_data[x:]
        y_train = y_data[0:x]
        y_test = y_data[x:]
        return x_train, x_test, y_train, y_test

    def plotConfusionMatrix(cm):
        cm.plot()
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def plotErrors(vector_error):
        fig, ax = plt.subplots()
        plt.title('Errors in training')
        plt.xlabel('Epochs')
        plt.ylabel('Errors')
        x = [range(len(vector_error))]
        ax.scatter(x, vector_error, marker='o', color=[0.00, 0.45, 0.74])
        ax.grid(True)
        plt.show()

    def inverse_transform(classes, n_classes):
        k = n_classes.shape[0]

        y_new = []
        for j in classes:
            for i in range(k):
                if np.array_equal(j, n_classes[i]):
                    y_new.append(i+1)
                    break
        return y_new
    
    def transform(classes):
        n_classes = np.unique(classes)
        k = len(n_classes)
        iden = np.identity(k)[::-1]
        
        classes_new = []
        for i in classes:
            for index, j in enumerate(n_classes):
                if np.array_equal(i, j):
                    aux = list(iden[index])
                    classes_new.append(aux)
        return np.array(classes_new)

    def plotColorMap(x_train, x_test, y_train, predict):

        color1_x = []
        color1_y = []
        color2_x = []
        color2_y = []
        color3_x = []
        color3_y = []
        for i in np.arange(0,1.0,0.005):
            for j in np.arange(0,1.0,0.005):
                xi = np.array([-1, i, j])
                y = predict(xi)
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
                    print('Error color!\n')
        
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
                print('Error!\n')
        train1 = x_train[i]
        train2 = x_train[j]
        train3 = x_train[k]

        fig, ax = plt.subplots()
        plt.title('Perceptron Color Map')
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

