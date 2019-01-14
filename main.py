from train import Model
import sys

data_name = sys.argv[1]
model = Model(data_name)
model.train()
