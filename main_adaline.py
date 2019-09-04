# -*- coding: utf-8 -*-
from Utils.dataset import *
from Utils.utils import *
from Adaline import Adaline

def main():
    # Import data
    x_data, y_data = artificial2D()

    model = Adaline.Adaline(x_data, y_data)
    model.adaline()

if __name__ == '__main__':
    main()