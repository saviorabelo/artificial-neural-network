# -*- coding: utf-8 -*-
from Utils.dataset import Data as data
from RadialBasisFunctionRegression import RBF as rbf

def main():
    # Artificial Database
    x_data, y_data = data.artificialSeno()

    #Database
    #x_data, y_data = data.abalone()
    #x_data, y_data = data.carFuel()

    model = rbf.RBF(x_data, y_data)
    model.execute()

if __name__ == '__main__':
    main()