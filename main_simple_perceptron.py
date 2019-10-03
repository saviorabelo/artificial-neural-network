# -*- coding: utf-8 -*-
from Utils.dataset import Data as data
from SimplePerceptron import Perceptron as ps

def main():

    x_data, y_data = data.irisFlowerBinary('Iris-setosa')
    #x_data, y_data = data.irisFlowerBinary('Iris-versicolor')
    #x_data, y_data = data.irisFlowerBinary('Iris-virginica')
    #x_data, y_data = data.vertebralColumnBinary()
    #x_data, y_data = data.dermatologyBinary()
    #x_data, y_data = data.cancerBinary()

    # Artificial
    #x_data, y_data = data.artificial2C()
    #x_data, y_data = data.artificialAND()
    #x_data, y_data = data.artificialOR()

    model = ps.Perceptron(x_data, y_data)
    model.perceptron()
    #model.plotColorMap()


if __name__ == '__main__':
    main()