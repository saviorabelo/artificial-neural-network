import random
import numpy as np
from matplotlib import pyplot as plt

class Util:
    def insertBias(data_):
        data = np.copy(data_)
        (m, _) = data.shape
        bias = -1 * np.ones((m,1))
        data = np.concatenate((bias, data), axis=1)
        return data
        
    def normalizeData(data_):
        data = np.copy(data_)
        max_ = data.max(axis=0)
        min_ = data.min(axis=0)
        data = (data - min_) / (max_ - min_)
        return data

    def shuffleData(x_, y_):
        x = np.copy(x_)
        y = np.copy(y_)
        data = list(zip(x, y))
        random.shuffle(data)
        x_aux, y_aux = zip(*data)
        x_aux, y_aux = np.array(x_aux), np.array(y_aux)
        return x_aux, y_aux

    def splitData(x_data_, y_data_, train_size):
        x_data = np.copy(x_data_)
        y_data = np.copy(y_data_)
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
    
    def plotEta(eta_vector):
        fig, ax = plt.subplots()
        plt.title('Eta in training')
        plt.xlabel('Epochs')
        plt.ylabel('Eta')
        x = [range(len(eta_vector))]
        ax.scatter(x, eta_vector, marker='o', color=[0.00, 0.45, 0.74])
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
