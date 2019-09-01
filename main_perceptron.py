# -*- coding: utf-8 -*-
from Utils.dataset import *
from Utils.utils import *
from Perceptron import Perceptron as PS

def main():
    #
    #attribute = 'Iris-setosa'
    #attribute = 'Iris-versicolor'
    #attribute = 'Iris-virginica'
    #x_data, y_data = irisFlowerBinary(attribute)

    #x_data, y_data = vertebralColumnBinary()
    #x_data, y_data = dermatologyBinary()
    x_data, y_data = cancerBinary()

    # Artificial
    #x_data, y_data = artificial2C()
    #x_data, y_data = artificialAND()
    #x_data, y_data = artificialOR()

    ps = PS.Perceptron(x_data, y_data)
    ps.perceptron()
    #ps.plotColorMap()


if __name__ == '__main__':
    main()