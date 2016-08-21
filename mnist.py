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
batch_losses = []
histories = []
test_final = []
for i in xrange(2):
	out = run_model(n_batch, n_in, n_layer, n_out, n_epoch,
		        p, -1,
	      	        X_train, Y_train, X_test, Y_test)

	batch_losses.append(out[0])
	histories.append(out[1])
	test_final.append(out[2])

output = {"batch_loss" : batch_losses,
		  "histories" : histories,
		  "test_final" : test_final}

with open("new_model.pkl", "wb") as f:
	cPickle.dump(output, f)

