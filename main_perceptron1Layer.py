# -*- coding: utf-8 -*-
from Utils.dataset import *
from Utils.utils import *
from Perceptron1Layer import Perceptron1Layer as PS

def main():

    x_data, y_data = artificial3C()

    ps = PS.Perceptron(x_data, y_data, normalize=False)
    ps.perceptron()
    ps.plotColorMap()

if __name__ == '__main__':
    main()