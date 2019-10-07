# -*- coding: utf-8 -*-
from Utils.dataset import Data as data
from MultilayerPerceptron import Perceptron as mlp

def main():

    # Database
    x_data, y_data = data.artificial3C()
    #x_data, y_data = data.irisFlower()
    #x_data, y_data = data.vertebralColumn()

    model = mlp.Perceptron(x_data, y_data, activation='logistic', hidden_layer=5)
    model.execute()

if __name__ == '__main__':
    main()