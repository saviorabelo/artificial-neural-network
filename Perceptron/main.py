# -*- coding: utf-8 -*-
from dataset import *
import Perceptron as ps

def main():
    # import dataset
    attribute = 'Iris-setosa'
    x_data, y_data = irisFlowerBinary(attribute)

    p = ps.Perceptron(x_data, y_data)
    p.perceptron()

if __name__ == '__main__':
    main()