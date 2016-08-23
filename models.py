# dropout_masks can take values in {0, 1, 2, -1}:
#


from keras.callbacks import Callback
from keras import backend as K
from keras.layers import Input, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
import time
import numpy as np

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

def evaluate(model, X, Y, n_batch, n_mc):
	# model is the trained model for evaluation
	# X, Y are test inputs and outputs respectively
	# n_batch is batch_size for prediction
	# n_mc is number of forward passes
	#	note that for models with 2 forward passes, 1 prediction already avgs 2 passes

	pred = None
	for i in xrange(n_mc):
		softmaxs = model.predict(X, batch_size=n_batch)
		if pred is None:
			pred = softmaxs / n_mc
		else:
			pred += softmaxs / n_mc

	return K.categorical_crossentropy(pred, Y).eval().mean()


# def evaluate(model, X, Y, n_batch, n_mc, get_final_output):
# 	def evaluate_batch(Xb, Yb):
# 		pred = None
# 		for i in xrange(n_mc):
# 			# learning_phase is 1 for train mode
# 			s = get_final_output([Xb, 1])
# 		 	if pred is None:
# 		 		pred = s / n_mc
# 		 	else:
# 		 		pred += s / n_mc
# 		# Xb_rep = np.tile(Xb, (n_mc, 1))
# 		# preds = get_final_output([Xb_rep, 1])
# 		# pred = preds.reshape(n_mc, len(Yb), Yb.shape[1]).mean(axis=0)
# 		return K.categorical_crossentropy(pred, Yb).eval().mean()

# 	batches = len(X) / n_batch
# 	score = None
# 	for batch_ind in xrange(batches):
# 		print batch_ind
# 		batch = range(batch_ind*n_batch, (batch_ind+1)*n_batch)
# 		score_batch = evaluate_batch(X[batch], Y[batch])
# 		if score is None:
# 			score = score_batch / batches
# 		else:
# 			score += score_batch / batches

# 	return score	

def apply_layers(input, layers):
	output = input
	for layer in layers:
		output = layer(output)

	return output


def define_model(n_in, n_layer, n_out, p, dropout_flag, batch_norm):
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

	# x = Input(batch_shape=(n_batch, n_in))
	x = Input(shape=(n_in,))
	maskx = K.dropout(K.ones((n_in,)), p)
	# print((1. / p - maskx).eval())

	layer1 = Dense(n_layer, activation='relu')
	batchnorm1 = BatchNormalization()
	mask1 = K.dropout(K.ones((n_layer,)), p)

	layer2 = Dense(n_layer, activation='relu')
	batchnorm2 = BatchNormalization()
	mask2 = K.dropout(K.ones((n_layer,)), p)

	layer3 = Dense(n_layer, activation='relu')
	batchnorm3 = BatchNormalization()
	mask3 = K.dropout(K.ones((n_layer,)), p)

	softmax_layer = Dense(n_out, activation='softmax')

	if dropout_flag == -1:
		maskx2 = (1. / p - maskx)
		mask4 = (1. / p - mask1)
		mask5 = (1. / p - mask2)
		mask6 = (1. / p - mask3)
	elif dropout_flag == 2:
		maskx2 = K.dropout(K.ones((n_in)), p)
		mask4 = K.dropout(K.ones((n_layer,)), p)
		mask5 = K.dropout(K.ones((n_layer,)), p)
		mask6 = K.dropout(K.ones((n_layer,)), p)

	dropout_in = dropout_layer1 = dropout_layer2 \
		= dropout_layer3 = dropout_layer4 = None
	if dropout_flag != 0:
		dropout_in = Lambda(lambda x: x * maskx, output_shape=(n_in,))
		dropout_layer1 = Lambda(lambda x: x * mask1, output_shape = (n_layer,))
		dropout_layer2 = Lambda(lambda x: x * mask2, output_shape = (n_layer,))
		dropout_layer3 = Lambda(lambda x: x * mask3, output_shape = (n_layer,))

	if dropout_flag == -1 or dropout_flag == 2:
		dropout_neg_in = Lambda(lambda x: x * maskx2, output_shape=(n_in,))
		dropout_neg_layer1 = Lambda(lambda x: x * mask4, output_shape = (n_layer,))
		dropout_neg_layer2 = Lambda(lambda x: x * mask5, output_shape = (n_layer,))
		dropout_neg_layer3 = Lambda(lambda x: x * mask6, output_shape = (n_layer,))

	# apply model for pass 1
	if batch_norm and dropout_flag != 0:
		layers = [dropout_in, layer1, batchnorm1,
				  dropout_layer1, layer2, batchnorm2,
				  dropout_layer2, layer3, batchnorm3,
				  dropout_layer3, softmax_layer]
	elif batch_norm and dropout_flag == 0:
		layers = [layer1, batchnorm1,
				  layer2, batchnorm2,
				  layer3, batchnorm3,
				  softmax_layer]
	elif dropout_flag != 0:
		layers = [dropout_in, layer1,
				  dropout_layer1, layer2,
				  dropout_layer2, layer3,
				  dropout_layer3, softmax_layer]
	else:
		layers = [layer1, layer2, layer3, softmax_layer]

	out1 = apply_layers(x, layers)

	# if necessary apply model for pass 2
	if dropout_flag == -1  or dropout_flag == 2:
		if batch_norm:
			layers = [dropout_neg_in, layer1, batchnorm1,
					  dropout_neg_layer1, layer2, batchnorm2,
					  dropout_neg_layer2, layer3, batchnorm3,
					  dropout_neg_layer3, softmax_layer]
		else:
			layers = [dropout_neg_in, layer1,
					  dropout_neg_layer1, layer2,
					  dropout_neg_layer2, layer3,
					  dropout_neg_layer3, softmax_layer]

		out2 = apply_layers(x, layers)

		avg = Lambda(lambda args: 0.5 * (args[0] + args[1]), output_shape=(n_out,))
		prediction = avg([out1, out2])
	else:
		prediction = out1

	model_stoch = Model(input=x, output=prediction)

	# define non-stoch dropout model for testing
	if batch_norm:
		layers = [layer1, batchnorm1,
				  layer2, batchnorm2,
				  layer3, batchnorm3,
				  softmax_layer]
	else:
		layers = [layer1, layer2, layer3, softmax_layer]

	prediction_non_stoch = apply_layers(x, layers)
	model_non_stoch = Model(input=x, output=prediction_non_stoch)

	return model_stoch, model_non_stoch


def run_model(n_in, n_layer, n_out, p, dropout_flag, batch_norm,
	n_batch, n_epoch, n_mc, X_train, Y_train, X_test, Y_test, test_n_batch):

	model_stoch, model_non_stoch = define_model(n_in, n_layer, n_out,
									            p, dropout_flag, batch_norm)

	model_stoch.compile(optimizer='adam',
	                    loss='categorical_crossentropy',
	                    metrics=['accuracy'])

	model_non_stoch.compile(optimizer='adam',
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
	test_batches.add(train_batches)

	# train the model
	for b in xrange(train_batches):
		batch_ind = b % batches_per_epoch
		batch = range(batch_ind*n_batch, (batch_ind+1)*n_batch)
		start = time.time()
		train_loss = model_stoch.fit(X_train[batch], Y_train[batch],
			                         verbose=0, batch_size=n_batch, nb_epoch=1)
		t = time.time() - start

		train_losses.append(train_loss.history['loss'][0])	
		train_times.append(t)

		if b+1 in test_batches:
			import pdb
			pdb.set_trace()

			print b+1
			# ind = np.arange(len(X_test))
			# np.random.shuffle(ind)
			test_loss = evaluate(model_stoch, X_test, Y_test,
								 test_n_batch, n_mc)
			test_losses_stoch.append(test_loss)

			test_loss = model_non_stoch.evaluate(X_test, Y_test,
												 batch_size=test_n_batch)
			test_losses_non_stoch.append(test_loss[0])	


	return train_losses, test_losses_stoch, test_losses_non_stoch, train_times
