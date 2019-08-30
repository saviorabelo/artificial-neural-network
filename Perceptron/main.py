# -*- coding: utf-8 -*-
from dataset import *
import Perceptron as PS

def main():
    # Iris
    #attribute = 'Iris-setosa'
    #attribute = 'Iris-versicolor'
    #attribute = 'Iris-virginica'
    #x_data, y_data = irisFlowerBinary(attribute)

    # Column
    #x_data, y_data = vertebralColumnBinary()

    # Artificial
    x_data, y_data = artificialAND()

    ps = PS.Perceptron(x_data, y_data)
    ps.perceptron()


if __name__ == '__main__':
    main()