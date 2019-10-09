# -*- coding: utf-8 -*-
from Utils.dataset import Data as data
from MultilayerPerceptron import Perceptron as mlp

def main():
    # Artificial Database
    #x_data, y_data = data.artificial3C()
    #x_data, y_data = data.artificialAND()
    #x_data, y_data = data.artificialOR()
    #x_data, y_data = data.artificialXOR()
    #x_data, y_data = data.artificialMoons()
    #x_data, y_data = data.artificialCircles()

    # Database
    #x_data, y_data = data.irisFlower()
    #x_data, y_data = data.vertebralColumn()
    #x_data, y_data = data.dermatology()
    x_data, y_data = data.cancer()

    # logistic tanh
    model = mlp.Perceptron(x_data, y_data, activation='logistic', g_search=False, hidden_layer=2)
    model.execute()

if __name__ == '__main__':
    main()