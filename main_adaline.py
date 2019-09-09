# -*- coding: utf-8 -*-
from Utils.dataset import *
from Utils.utils import *
from Adaline import Adaline as ad

def main():
    # Import data
    #x_data, y_data = artificial2D()
    x_data, y_data = artificial3D()

    model = ad.Adaline(x_data, y_data)
    model.adaline()
    model.plotColorMap()

if __name__ == '__main__':
    main()