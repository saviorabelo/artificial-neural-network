import random
import numpy as np

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
