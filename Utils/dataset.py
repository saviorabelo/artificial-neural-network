import random
import numpy as np
import pandas as pd
from Utils.utils import Util as util
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons, make_circles

class Data:
    def artificialMoons():
        X, y  = make_moons(n_samples=300, noise=0.05,random_state=10)

        return X, util.transform(y)
    
    def artificialCircles():
        X, y  = make_circles(n_samples=300, factor=.5, noise=.05)

        return X, util.transform(y)

    def artificial2D():
        a = 2
        b = 10
        n = 100
        noise = 0.3

        x = np.linspace(0, 1, n).reshape((-1, 1))

        y = a*x + b
        for i in range(n):
            y[i] += random.uniform(-noise, noise)

        return x, y
    
    def artificialSeno():
        n = 300
        noise = 0.2
        x = np.linspace(-5, 5, n).reshape((-1, 1))

        y = np.sin(x)+2
        for i in range(n):
            y[i] += random.uniform(-noise, noise)
        
        return x, y

    def artificial3D():
        a = 2
        b = 4
        c = 5
        n = 300

        x = np.linspace(0, 1, n).reshape((-1, 1))
        
        y = np.linspace(0, 1, n).reshape((-1, 1))
        for i in range(n):
            y[i] += random.uniform(-1, 1)
        
        z = a*x + b*y + c
        for i in range(n):
            z[i] += random.uniform(-1, 1)
        
        aux = np.concatenate((x, y), axis=1)
        return aux, z

    def artificialAND():
        # Data size 4*n*n
        n = 7
        x1 = [[random.uniform(0.2, 0.4), random.uniform(0.2, 0.4)] for _ in range(n) for _ in range(n)]
        x2 = [[random.uniform(0.2, 0.4), random.uniform(0.6, 0.8)] for _ in range(n) for _ in range(n)]
        x3 = [[random.uniform(0.6, 0.8), random.uniform(0.2, 0.4)] for _ in range(n) for _ in range(n)]
        x4 = [[random.uniform(0.6, 0.8), random.uniform(0.6, 0.8)] for _ in range(n) for _ in range(n)]
        x = np.concatenate((x1, x2), axis=0)
        x = np.concatenate((x, x3), axis=0)
        x = np.concatenate((x, x4), axis=0)

        y123 = [[0,1] for _ in range(3*n*n)]
        y4 = [[1,0] for _ in range(n*n)]
        y = np.concatenate((y123, y4), axis=0)

        return x, y

    def artificialOR():
        # Data size 4*n*n
        n = 7
        x1 = [[random.uniform(0.2, 0.4), random.uniform(0.2, 0.4)] for _ in range(n) for _ in range(n)]
        x2 = [[random.uniform(0.2, 0.4), random.uniform(0.6, 0.8)] for _ in range(n) for _ in range(n)]
        x3 = [[random.uniform(0.6, 0.8), random.uniform(0.2, 0.4)] for _ in range(n) for _ in range(n)]
        x4 = [[random.uniform(0.6, 0.8), random.uniform(0.6, 0.8)] for _ in range(n) for _ in range(n)]
        x = np.concatenate((x1, x2), axis=0)
        x = np.concatenate((x, x3), axis=0)
        x = np.concatenate((x, x4), axis=0)

        y1 = [[0,1] for _ in range(n*n)]
        y234 = [[1,0] for _ in range(3*n*n)]
        
        y = np.concatenate((y1, y234), axis=0)

        return x, y
    
    def artificialXOR():
        # Data size 4*n*n
        n = 7
        x1 = [[random.uniform(0.2, 0.4), random.uniform(0.2, 0.4)] for _ in range(n) for _ in range(n)]
        x2 = [[random.uniform(0.6, 0.8), random.uniform(0.6, 0.8)] for _ in range(n) for _ in range(n)]
        x3 = [[random.uniform(0.2, 0.4), random.uniform(0.6, 0.8)] for _ in range(n) for _ in range(n)]
        x4 = [[random.uniform(0.6, 0.8), random.uniform(0.2, 0.4)] for _ in range(n) for _ in range(n)]
        x = np.concatenate((x1, x2), axis=0)
        x = np.concatenate((x, x3), axis=0)
        x = np.concatenate((x, x4), axis=0)

        y1 = [[0,1] for _ in range(2*n*n)]
        y234 = [[1,0] for _ in range(2*n*n)]
        
        y = np.concatenate((y1, y234), axis=0)

        return x, y

    def artificial2C():
        # Data size 2*n*n
        n = 7
        x1 = [[random.uniform(0.2, 0.4), random.uniform(0.2, 0.4)] for _ in range(n) for _ in range(n)]
        x2 = [[random.uniform(0.6, 0.8), random.uniform(0.6, 0.8)] for _ in range(n) for _ in range(n)]
        x = np.concatenate((x1, x2), axis=0)

        y1 = [[0] for _ in range(n*n)]
        y2 = [[1] for _ in range(n*n)]
        y = np.concatenate((y1, y2), axis=0)

        return x, y

    def artificial3C():
        # Data size 3*n*n
        n = 7

        x1 = [[random.uniform(0.2, 0.4), random.uniform(0.2, 0.4)] for _ in range(n) for _ in range(n)]
        x2 = [[random.uniform(0.4, 0.6), random.uniform(0.6, 0.8)] for _ in range(n) for _ in range(n)]
        x3 = [[random.uniform(0.6, 0.8), random.uniform(0.2, 0.4)] for _ in range(n) for _ in range(n)]
        x = np.concatenate((x1, x2), axis=0)
        x = np.concatenate((x, x3), axis=0)

        y1 = [[0,0,1] for _ in range(n*n)]
        y2 = [[0,1,0] for _ in range(n*n)]
        y3 = [[1,0,0] for _ in range(n*n)]
        y = np.concatenate((y1, y2), axis=0)
        y = np.concatenate((y, y3), axis=0)

        return x, y

    def irisFlower():
        dataset = pd.read_csv('Datasets/iris.data', sep=',')
        classe = dataset['Classe']
        x = dataset.drop(['Classe'], axis=1)
        y = util.transform(classe)

        # Remove column
        #x = x.drop(['Petal_Length'], axis=1)
        #x = x.drop(['Petal_Width'], axis=1)

        return np.array(x), y

    def irisFlowerBinary(attribute):
        dataset = pd.read_csv('Datasets/iris.data', sep=',')
        classe = dataset['Classe']
        x = dataset.drop(['Classe'], axis=1)
        
        # Remove column
        #x = x.drop(['Petal_Length'], axis=1)
        #x = x.drop(['Petal_Width'], axis=1)
        
        y = []
        for i in classe:
            if(i == attribute):
                y.append([1])
            else:
                y.append([0])

        return np.array(x), np.array(y)

    def vertebralColumnBinary():
        dataset = pd.read_csv('Datasets/column_2C.dat', sep=' ')
        classe = dataset['Classe']
        x = dataset.drop(['Classe'], axis=1)

        # Remove column
        #x = x.drop(['lumbar_lordosis_angle'], axis=1)
        #x = x.drop(['sacral_slope'], axis=1)
        #x = x.drop(['pelvic_radius'], axis=1)
        #x = x.drop(['grade_of_spondylolisthesis'], axis=1)

        y = []
        for i in classe:
            if(i == 'AB'):
                y.append([1])
            else:
                y.append([0])

        return np.array(x), np.array(y)

    def vertebralColumn():
        dataset = pd.read_csv('Datasets/column_3C.dat', sep=' ')
        classe = dataset['Classe']
        x = dataset.drop(['Classe'], axis=1)
        y = util.transform(classe)

        # Remove column
        #x = x.drop(['lumbar_lordosis_angle'], axis=1)
        #x = x.drop(['sacral_slope'], axis=1)
        #x = x.drop(['pelvic_radius'], axis=1)
        #x = x.drop(['grade_of_spondylolisthesis'], axis=1)

        return np.array(x), y
    
    def dermatology():
        dataset = pd.read_csv('Datasets/dermatology.dat', sep=',')
        classe = dataset['Classe']
        x = dataset.drop(['Classe'], axis=1)
        y = util.transform(classe)

        return np.array(x), y

    def dermatologyBinary():
        dataset = pd.read_csv('Datasets/dermatology.dat', sep=',')
        classe = dataset['Classe']
        x = dataset.drop(['Classe'], axis=1)

        y = []
        for i in classe:
            if(i == 1):
                y.append([1])
            else:
                y.append([0])

        return np.array(x), np.array(y)
    
    def cancer():
        dataset = pd.read_csv('Datasets/wdbc.data', sep=',')
        classe = dataset['Classe']
        x = dataset.drop(['Classe'], axis=1)
        y = util.transform(classe)

        return np.array(x), y

    def cancerBinary():
        dataset = pd.read_csv('Datasets/wdbc.data', sep=',')
        classe = dataset['Classe']
        x = dataset.drop(['Classe'], axis=1)
        x = x.drop(['ID'], axis=1)

        y = []
        for i in classe:
            if(i == 'M'):
                y.append([1])
            else:
                y.append([0])

        return np.array(x), np.array(y)
