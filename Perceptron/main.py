# -*- coding: utf-8 -*-
from dataset import *
import Perceptron as PS

def main():
    # import dataset
    attribute = 'Iris-setosa'
    x_data, y_data = irisFlowerBinary(attribute)

    ps = PS.Perceptron(x_data, y_data)
    ps.perceptron()

if __name__ == '__main__':
    main()