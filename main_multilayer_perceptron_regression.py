# -*- coding: utf-8 -*-
from Utils.dataset import Data as data
from MultilayerPerceptronRegression import Perceptron as mlp

def main():
    # Artificial Database
    x_data, y_data = data.artificialSeno()

    model = mlp.Perceptron(x_data, y_data, activation='logistic', hidden_layer=10)
    model.execute()

if __name__ == '__main__':
    main()