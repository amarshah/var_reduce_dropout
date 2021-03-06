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
import numpy as np

from models import run_model
import argparse

parser = argparse.ArgumentParser(description="training a model")
parser.add_argument("dropout_flag", type=int, default=-1)
parser.add_argument("n_epoch", type=int, default=1)
parser.add_argument("n_layer", type=int, default=100)
parser.add_argument("n_runs", type=int, default=20)
parser.add_argument("n_mc", type=int, default=10)
parser.add_argument("batch_norm", type=bool, default=False)
parser.add_argument("save_file", type=str, default="save.pkl")

args = parser.parse_args()
d = vars(args)
dropout_flag = d["dropout_flag"]
n_epoch = d["n_epoch"]
n_layer = d["n_layer"]
n_runs = d["n_runs"]
n_mc = d["n_mc"]
batch_norm = d["batch_norm"]=="True"
save_file = d["save_file"]

n_batch = 20
test_n_batch = 20 # TODO: batch size has to be a divisor of len(X_test)

n_in = 784
n_out = 10
n_classes = 10
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
# test_losses_stoch = []
test_mc_probs = []
test_losses_non_stoch = []
train_times = []
test_batches = []
for i in xrange(n_runs):
	np.random.seed(i)
	perm = np.random.permutation(X_train.shape[0])
	perm_test = np.random.permutation(X_test.shape[0])

	out = run_model(n_in, n_layer, n_out, p, dropout_flag, batch_norm,
		n_batch, n_epoch, n_mc,
		X_train[perm], Y_train[perm],
		X_test[perm_test[:1000]], Y_test[perm_test[:1000]], test_n_batch)

	train_losses.append(out[0])
	# test_losses_stoch.append(out[1])
	test_mc_probs.append(out[1])
	test_losses_non_stoch.append(out[2])
	train_times.append(out[3])
	test_batches.append(out[4])

	output = {"train_losses" : train_losses,
			  "test_mc_probs" : test_mc_probs,	
			  "test_losses_non_stoch" : test_losses_non_stoch,	
			  "train_times" : train_times,
			  "test_batches" : test_batches}

	with open(save_file, "wb") as f:
		cPickle.dump(output, f)
