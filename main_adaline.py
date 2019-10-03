# -*- coding: utf-8 -*-
from Adaline import Adaline as ad
from Utils.dataset import Data as data

def main():
    # Import data
    #x_data, y_data = data.artificial2D()
    x_data, y_data = data.artificial3D()

    model = ad.Adaline(x_data, y_data)
    model.adaline()
    #model.plotColorMap()

if __name__ == '__main__':
    main()
