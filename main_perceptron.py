# -*- coding: utf-8 -*-
from Utils.dataset import Data as data
from Perceptron import Perceptron as pc

def main():

    # Database
    x_data, y_data = data.artificial3C()
    #x_data, y_data = data.irisFlower()
    #x_data, y_data = data.vertebralColumn()

    model = pc.Perceptron(x_data, y_data, activation='logistic')
    model.execute()

if __name__ == '__main__':
    main()