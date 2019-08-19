# -*- coding: utf-8 -*-
from Utils import dataset as db
import Perceptron as ps

# import dataset
database = db.dataset()
x_data, y_data = database.irisFlowerBinary()

ps = ps.Perceptron(x_data, y_data)
ps.train()