# ICLR Best paper award 2017 code
#
# All rights reserved for LBL inc.
# 
# Dropoot training with variance reduction masks


from keras import backend as K
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import Callback
import cPickle

from models import run_model
import argparse

parser = argparse.ArgumentParser(
    description="training a model")
parser.add_argument("model_flag", type=int, default=-1)
parser.add_argument("n_runs", type=int, default=100)
parser.add_argument("n_mc", type=int, default=10)
parser.add_argument("save_file")

args = parser.parse_args()
d = vars(args)
model_flag = d["model_flag"]
save_file = d["save_file"]
n_runs = d["n_runs"]
n_mc = d["n_mc"]
# model_flag = -1
# save_file = "new_model.pkl"
# n_runs = 100

n_batch = 20
n_in = 784
n_out = 10
n_layer = 1024
n_classes = 10
n_epoch = 1
p = 0.5

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

###################################################################
train_losses = []
test_losses = []
train_times = []
for i in xrange(n_runs):
	out = run_model(n_batch, n_in, n_layer, n_out, n_epoch,
		p, model_flag, n_mc,
		X_train, Y_train, X_test, Y_test)

	train_losses.append(out[0])
	test_losses.append(out[1])
	train_times.append(out[2])

output = {"train_losses" : train_losses,
		  "test_losses" : test_losses,
		  "train_times" : train_times}

with open(save_file, "wb") as f:
	cPickle.dump(output, f)

