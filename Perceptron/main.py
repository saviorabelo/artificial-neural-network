# -*- coding: utf-8 -*-
from sklearn import datasets

dataset = datasets.load_iris()

X = dataset.data
Y = dataset.target

print(Y)