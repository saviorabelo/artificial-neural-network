# -*- coding: utf-8 -*-
from Utils.dataset import Data as data
from ExtremeLearningMachineRegression import ELM as elm

def main():
    # Artificial Database
    x_data, y_data = data.artificialSeno()

    #Database
    #x_data, y_data = data.abalone()
    #x_data, y_data = data.carFuel()

    model = elm.ELM(x_data, y_data)
    model.execute()

if __name__ == '__main__':
    main()