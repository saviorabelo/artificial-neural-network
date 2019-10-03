# -*- coding: utf-8 -*-
import numpy as np
from Utils.dataset import Data as data
from Perceptron import Perceptron as pc

def main():

    #x_data, y_data = data.artificial3C()
    #x_data, y_data = data.irisFlower()
    x_data, y_data = data.vertebralColumn()

    ps = pc.Perceptron(x_data, y_data, activation='logistic')
    ps.execute()

if __name__ == '__main__':
    main()