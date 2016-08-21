# dropout_masks can be 1, 2, or -1 for control variate

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))


def run_model(n_batch, n_in, n_layer, n_out, n_epoch,
	          p, dropout_masks = -1,
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

	avg = Lambda(lambda args: 0.5 * (args[0] + args[1]), output_shape=(n_out,))
	prediction = avg([out1, out2])

	model = Model(input=x, output=prediction)

	model.compile(optimizer='adam',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])

	history = LossHistory()
	epoch_history = model.fit(X_train, Y_train,
    	      				  batch_size=n_batch, nb_epoch=n_epoch,
        	  				  verbose=0, validation_data=(X_test, Y_test),
          					  callbacks=[history])

	batch_loss = history.losses
	history = epoch_history.history
	score = model.evaluate(X_test, Y_test, batch_size=n_batch, verbose=0)

	return batch_loss, history, score
