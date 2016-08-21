# dropout_masks can be 1, 2, or -1 for control variate

from keras.callbacks import Callback
from keras import backend as K
from keras.layers import Input, Dense, Lambda
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
	get_final_output = K.function([model.layers[0].input, K.learning_phase()], 
								   model.layers[-1].output)

	def evaluate_batch(Xb, Yb):
		# pred = None
		# for i in xrange(n_mc):
		# 	# learning_phase is 1 for train mode
		# 	s = get_final_output([Xb, 1])
		# 	if pred is None:
		# 		pred = s / n_mc
		# 	else:
		# 		pred += s / n_mc
		Xb_rep = np.tile(Xb, (n_mc, 1))
		preds = get_final_output([Xb_rep, 1])
		pred = preds.reshape(n_mc, len(Xb), Xb.shape[1]).sum(axis=0)
		return K.categorical_crossentropy(pred, Yb).eval().mean()

	batches = len(X) / n_batch
	score = None
	for batch_ind in xrange(batches):
		print batch_ind
		batch = range(batch_ind*n_batch, (batch_ind+1)*n_batch)
		score_batch = evaluate_batch(X[batch], Y[batch])
		if score is None:
			score = score_batch / batches
		else:
			score += score_batch / batches

	return score


def run_model(n_batch, n_in, n_layer, n_out, n_epoch,
	      p, dropout_masks,
	      X_train, Y_train, X_test, Y_test):

	# specify model
	x = Input(batch_shape=(n_batch, n_in))
	maskx = K.dropout(K.ones((n_batch, n_in)), p)
	# print((1. / p - maskx).eval())

	layer1 = Dense(1024, activation='relu')
	mask1 = K.dropout(K.ones((n_batch, n_layer)), p)

	layer2 = Dense(1024, activation='relu')
	mask2 = K.dropout(K.ones((n_batch, n_layer)), p)

	layer3 = Dense(1024, activation='relu')
	mask3 = K.dropout(K.ones((n_batch, n_layer)), p)

	layer4 = Dense(1024, activation='relu')
	mask4 = K.dropout(K.ones((n_batch, n_layer)), p)

	softmax_layer = Dense(n_out, activation='softmax')


	dropout_in = Lambda(lambda x: x * maskx, output_shape=(n_in,))
	dropout_layer1 = Lambda(lambda x: x * mask1, output_shape = (n_layer,))
	dropout_layer2 = Lambda(lambda x: x * mask2, output_shape = (n_layer,))
	dropout_layer3 = Lambda(lambda x: x * mask3, output_shape = (n_layer,))
	dropout_layer4 = Lambda(lambda x: x * mask4, output_shape = (n_layer,))

	if dropout_masks == -1:
		maskx2 = (1. / p - maskx)
		mask5 = (1. / p - mask1)
		mask6 = (1. / p - mask2)
		mask7 = (1. / p - mask3)
		mask8 = (1. / p - mask4)
	elif dropout_masks == 2:
		maskx2 = K.dropout(K.ones((n_batch, n_in)), p)
		mask5 = K.dropout(K.ones((n_batch, n_layer)), p)
		mask6 = K.dropout(K.ones((n_batch, n_layer)), p)
		mask7 = K.dropout(K.ones((n_batch, n_layer)), p)
		mask8 = K.dropout(K.ones((n_batch, n_layer)), p)
	elif dropout_masks == 1:
		maskx2 = maskx
		mask5 = mask1
		mask6 = mask2
		mask7 = mask3
		mask8 = mask4

	dropout_neg_in = Lambda(lambda x: x * maskx2, output_shape=(n_in,))
	dropout_neg_layer1 = Lambda(lambda x: x * mask5, output_shape = (n_layer,))
	dropout_neg_layer2 = Lambda(lambda x: x * mask6, output_shape = (n_layer,))
	dropout_neg_layer3 = Lambda(lambda x: x * mask7, output_shape = (n_layer,))
	dropout_neg_layer4 = Lambda(lambda x: x * mask8, output_shape = (n_layer,))

	# apply model
	out1 = layer1(dropout_in(x))
	out1 = layer2(dropout_layer1(out1))
	out1 = layer3(dropout_layer2(out1))
	out1 = layer4(dropout_layer3(out1))
	out1 = softmax_layer(dropout_layer4(out1))

	out2 = layer1(dropout_neg_in(x))
	out2 = layer2(dropout_neg_layer1(out2))
	out2 = layer3(dropout_neg_layer2(out2))
	out2 = layer4(dropout_neg_layer3(out2))
	out2 = softmax_layer(dropout_neg_layer4(out2))

	if dropout_masks==1:
		prediction = out1
	else:
		avg = Lambda(lambda args: 0.5 * (args[0] + args[1]), output_shape=(n_out,))
		prediction = avg([out1, out2])

	model = Model(input=x, output=prediction)

	model.compile(optimizer='adam',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])

	# history = LossHistory()
	# epoch_history = model.fit(X_train, Y_train,
	# 	batch_size=n_batch, nb_epoch=n_epoch,
	# 	verbose=0, validation_data=(X_test, Y_test),
	# 	callbacks=[history])

	# batch_loss = history.losses
	# history = epoch_history.history
	# score = model.evaluate(X_test, Y_test, batch_size=n_batch, verbose=0)

	# initial train and test loss
	out = model.evaluate(X_train, Y_train, batch_size=n_batch)
	train_losses = [out[0]]
	out = model.evaluate(X_test, Y_test, batch_size=n_batch)
	test_losses = [out[0]]
	train_times = [0]

	for e in xrange(n_epoch):
		for batch_ind in xrange(len(X_train) / n_batch):
			batch = range(batch_ind*n_batch, (batch_ind+1)*n_batch)
			start = time.time()
			train_loss = model.fit(X_train[batch], Y_train[batch], batch_size=n_batch, nb_epoch=1)
			t = time.time() - start
	
			train_losses.append(train_loss.history['loss'][0])	
			train_times.append(t)

			if (batch_ind+1) % 10 == 0:
				test_loss = evaluate(model, X_test, Y_test, n_batch, 10)
				test_losses.append(test_loss)	
				# test_loss = model.evaluate(X_test, Y_test, batch_size=n_batch)
				# test_losses.append(test_loss[0])	

	return train_losses, test_losses, train_times
