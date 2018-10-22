from train import *
import sys

data_name = sys.argv[1]
model = Model(data_name)
model.train()