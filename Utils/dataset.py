import numpy as np
import pandas as pd

class dataset:

    def __init__(self):
        self.x = []
        self.d = []

    def irisFlower(self):
        dataset = pd.read_csv('Datasets/iris.data', sep=',')
        flowers = np.array(dataset['Classe'])

        self.x = np.array(dataset.drop(['Classe'], axis=1))
        
        for flower in flowers:
            if(flower == 'Iris-setosa'):
                self.d.append([1, 0, 0])
            elif(flower == 'Iris-versicolor'):
                self.d.append([0, 1, 0])
            elif(flower == 'Iris-virginica'):
                self.d.append([0, 0, 1])
            else:
                print('Error!')

        return np.array(self.x), np.array(self.d)

#End dataset