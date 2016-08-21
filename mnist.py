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

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))


n_batch = 20
n_in = 784
n_out = 10
n_layer = 1024
n_classes = 10
n_epoch = 4
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

dropout_neg_in = Lambda(lambda x: x * (1. / p - maskx), output_shape=(n_in,))
dropout_neg_layer1 = Lambda(lambda x: x * (1. / p - mask1), output_shape = (n_layer,))
dropout_neg_layer2 = Lambda(lambda x: x * (1. / p - mask2), output_shape = (n_layer,))
dropout_neg_layer3 = Lambda(lambda x: x * (1. / p - mask3), output_shape = (n_layer,))
dropout_neg_layer4 = Lambda(lambda x: x * (1. / p - mask4), output_shape = (n_layer,))


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
# prediction = out1
# prediction = out2

# this creates a model that includes
# the Input layer and three Dense layers
model = Model(input=x, output=prediction)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = LossHistory()
model.fit(X_train, Y_train,
          batch_size=n_batch, nb_epoch=n_epoch,
          verbose=1, validation_data=(X_test, Y_test),
          callbacks=[history])

score = model.evaluate(X_test, Y_test, batch_size=n_batch, verbose=0)

print(score)
import pdb
pdb.set_trace()

