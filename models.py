# dropout_masks can take values in {0, 1, 2, -1}:
#


from keras.callbacks import Callback
from keras import backend as K
from keras.layers import Input, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras.optimizers import Adam, SGD
import time
import numpy as np

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

def accuracy(y_pred, y_true):
	return np.mean(np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1))

def evaluate_stoch(model, X, Y, n_batch, n_mc):
	# model is the trained model for evaluation
	# X, Y are test inputs and outputs respectively
	# n_batch is batch_size for prediction
	# n_mc is number of forward passes
	#	note that for models with 2 forward passes, 1 prediction already avgs 2 passes

	pred = None
	for i in xrange(n_mc):
		softmaxs = model.predict(X, batch_size=n_batch, verbose=0)
		if pred is None:
			pred = softmaxs / n_mc
		else:
			pred += softmaxs / n_mc

	# return K.categorical_crossentropy(pred, Y).eval().mean()
	return accuracy(pred, Y)

def evaluate_non_stoch(model_non_stoch, model_stoch, X, Y, batch_norm, n_batch):
	if batch_norm:
		# need to set all the batch norm parameters; we assume 
		# batch normalisation layers' names start with bn
		stoch_bn_layers = []
		for stoch_layer in model_stoch.layers:
			if stoch_layer.name[:2] == "bn":
				stoch_bn_layers += [stoch_layer]
		deter_bn_layers = []
		for deter_layer in model_non_stoch.layers:
			if deter_layer.name[:2] == "bn":
				deter_bn_layers += [deter_layer]

		for (stoch_layer, deter_layer) in zip(stoch_bn_layers, deter_bn_layers):
			running_mean = stoch_layer.running_mean.get_value()
			running_std = stoch_layer.running_std.get_value()
			beta = stoch_layer.beta.get_value()
			gamma = stoch_layer.gamma.get_value()
			deter_layer.running_mean.set_value(running_mean)
			deter_layer.running_std.set_value(running_std)
			deter_layer.beta.set_value(beta)
			deter_layer.gamma.set_value(gamma)

	# test_loss = model_non_stoch.evaluate(X, Y, n_batch, verbose=0)
	# return test_loss[0]
	softmaxs = model_non_stoch.predict(X, batch_size=n_batch, verbose=0)
	return accuracy(softmaxs, Y)

def apply_layers(input, layers):
	output = input
	for layer in layers:
		output = layer(output)

	return output


def define_model(n_batch, n_in, n_layer, n_out, p, dropout_flag, batch_norm):
	# n_in is the input dim
	# n_layer is the number of hidden units in the hidden layers
	# n_out is the number of outputs of the softmax
	# dropout_flag takes value 0,1,2 or -1
	#	0,1,2 is the number of different masks to use per training batch
	#   -1 forces 2 masks per batch, where 1 mask is the mirror of the other
	#
	# the function outputs 2 models: 1 with stochastic dropout, 1 with no dropout
	# 	the stoch one has stoch dropout even at test time	
	#   the non-stoch one uses the typical dropout approximation

	# Naming convention: maskM_L where M is model # and L is layer #.
	x = Input(batch_shape=(n_batch, n_in))
	layer1 = Dense(n_layer, activation='relu')
	layer2 = Dense(n_layer, activation='relu')
	layer3 = Dense(n_layer, activation='relu')
	softmax_layer = Dense(n_out, activation='softmax')

	# TODO: add back n_batch for mask; you can do this using 
	# n_batch = T.iscalar('n_batch') and pass it at run time or 
	# using models with different batch sizes.
	mask1_x = K.dropout(K.ones((n_batch, n_in)), p)
	mask1_1 = K.dropout(K.ones((n_batch, n_layer)), p)
	mask1_2 = K.dropout(K.ones((n_batch, n_layer)), p)
	mask1_3 = K.dropout(K.ones((n_batch, n_layer)), p)
	if dropout_flag == -1:
		mask2_x = (1. / p - mask1_x)
		mask2_1 = (1. / p - mask1_1)
		mask2_2 = (1. / p - mask1_2)
		mask2_3 = (1. / p - mask1_3)
	elif dropout_flag == 2:
		mask2_x = K.dropout(K.ones((n_batch, n_in)), p)
		mask2_1 = K.dropout(K.ones((n_batch, n_layer)), p)
		mask2_2 = K.dropout(K.ones((n_batch, n_layer)), p)
		mask2_3 = K.dropout(K.ones((n_batch, n_layer)), p)

	def get_dropout_layer(mask, n, name):
		return Lambda(lambda x: x * mask, output_shape=(n,), name=name)
	if dropout_flag != 0:
		dropout1_x = get_dropout_layer(mask1_x, n_in, "drop1_x")
		dropout1_1 = get_dropout_layer(mask1_1, n_layer, "drop1_1")
		dropout1_2 = get_dropout_layer(mask1_2, n_layer, "drop1_2")
		dropout1_3 = get_dropout_layer(mask1_3, n_layer, "drop1_3")

	if dropout_flag == -1 or dropout_flag == 2:
		dropout2_x = get_dropout_layer(mask2_x, n_in, "drop2_x")
		dropout2_1 = get_dropout_layer(mask2_1, n_layer, "drop2_1")
		dropout2_2 = get_dropout_layer(mask2_2, n_layer, "drop2_2")
		dropout2_3 = get_dropout_layer(mask2_3, n_layer, "drop2_3")

	# apply model for pass 1
	if batch_norm and dropout_flag != 0:
		layers = [dropout1_x, layer1, BatchNormalization(mode=2, name="bn1"),
				  dropout1_1, layer2, BatchNormalization(mode=2, name="bn2"),
				  dropout1_2, layer3, BatchNormalization(mode=2, name="bn3"),
				  dropout1_3, softmax_layer]
	elif batch_norm and dropout_flag == 0:
		layers = [layer1, BatchNormalization(mode=2, name="bn1"),
				  layer2, BatchNormalization(mode=2, name="bn2"),
				  layer3, BatchNormalization(mode=2, name="bn3"),
				  softmax_layer]
	elif not batch_norm and dropout_flag != 0:
		layers = [dropout1_x, layer1,
				  dropout1_1, layer2,
				  dropout1_2, layer3,
				  dropout1_3, softmax_layer]
	else:
		layers = [layer1, layer2, layer3, softmax_layer]

	out1 = apply_layers(x, layers)
	prediction = out1

	# if necessary apply model for pass 2
	if dropout_flag == -1  or dropout_flag == 2:
		if batch_norm:
			layers = [dropout2_x, layer1, BatchNormalization(mode=2),
					  dropout2_1, layer2, BatchNormalization(mode=2),
					  dropout2_2, layer3, BatchNormalization(mode=2),
					  dropout2_3, softmax_layer]
		else:
			layers = [dropout2_x, layer1,
					  dropout2_1, layer2,
					  dropout2_2, layer3,
					  dropout2_3, softmax_layer]

		out2 = apply_layers(x, layers)

		avg = Lambda(lambda args: 0.5 * (args[0] + args[1]), output_shape=(n_out,))
		# override prediction
		prediction = avg([out1, out2])

	model_stoch = Model(input=x, output=prediction)

	# define non-stoch dropout model for testing
	if batch_norm:
		layers = [layer1, BatchNormalization(mode=2, name="bn_non_stoch_1"),
				  layer2, BatchNormalization(mode=2, name="bn_non_stoch_2"),
				  layer3, BatchNormalization(mode=2, name="bn_non_stoch_3"),
				  softmax_layer]
	else:
		layers = [layer1, layer2, layer3, softmax_layer]

	prediction_non_stoch = apply_layers(x, layers)
	model_non_stoch = Model(input=x, output=prediction_non_stoch)

	return model_stoch, model_non_stoch


def run_model(n_in, n_layer, n_out, p, dropout_flag, batch_norm,
	n_batch, n_epoch, n_mc, X_train, Y_train, X_test, Y_test, test_n_batch):

	model_stoch, model_non_stoch = define_model(n_batch, n_in, n_layer, n_out,
									            p, dropout_flag, batch_norm)

	optimizer = Adam(lr=0.0002)  # SGD(lr=1e-6, momentum=0.9)
	model_stoch.compile(optimizer=optimizer,#"adam",
	                    loss='categorical_crossentropy',
	                    metrics=['accuracy'])

	model_non_stoch.compile(optimizer='sgd', 
				   			loss='categorical_crossentropy',
	                    	metrics=['accuracy'])

	# get_final_output = K.function([model_stoch.layers[0].input, K.learning_phase()], 
	# 							   model_stoch.layers[-1].output)

	# history = LossHistory()
	# epoch_history = model.fit(X_train, Y_train,
	# 	batch_size=n_batch, nb_epoch=n_epoch,
	# 	verbose=0, validation_data=(X_test, Y_test),
	# 	callbacks=[history])

	# batch_loss = history.losses
	# history = epoch_history.history
	# score = model.evaluate(X_test, Y_test, batch_size=n_batch, verbose=0)

	train_losses = []
	test_losses_stoch = []
	test_losses_non_stoch = []
	train_times = []

	# test at train batches spaced apart exponentially
	batches_per_epoch = len(X_train) / n_batch
	train_batches = batches_per_epoch * n_epoch
	test_batches = set()
	j = 1
	while j < train_batches - 1:
		test_batches.add(j)
		j *= 2
	test_batches.add(train_batches) # eval model at last batch

	# train the model
	for b in xrange(train_batches):
		batch_ind = b % batches_per_epoch
		batch = range(batch_ind*n_batch, (batch_ind+1)*n_batch)
		start = time.time()
		train_loss = model_stoch.fit(X_train[batch], Y_train[batch],
			                         verbose=0, batch_size=n_batch, nb_epoch=1)
		t = time.time() - start
		import pdb
		pdb.set_trace()

		train_losses.append(train_loss.history['loss'][0])	
		train_times.append(t)

		if b+1 in test_batches:
			print b+1
			# ind = np.arange(len(X_test))
			# np.random.shuffle(ind)
			test_loss = evaluate_stoch(model_stoch, X_test, Y_test, test_n_batch, n_mc)
			print(test_loss)
			test_losses_stoch.append(test_loss)

			test_loss = evaluate_non_stoch(model_non_stoch, model_stoch,
										   X_test, Y_test, batch_norm, test_n_batch)
			print(test_loss)
			test_losses_non_stoch.append(test_loss)	

	return train_losses, test_losses_stoch, test_losses_non_stoch, train_times
