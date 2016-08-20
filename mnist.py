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


n_batch = 20
n_in = 784
n_layer = 1024
n_classes = 10
n_epoch = 20
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

# this layer accepts a tuple (input and mask) and returns the product of the two.
dropout = Lambda(lambda args: args[0] * args[1], 
				 output_shape=lambda args: K.get_shape(args[0]))
#Example:
#out1 = dropout([x, maskx])

# specify model
x = Input(batch_shape=(n_batch, n_in))
maskx = K.dropout(K.ones((n_batch, n_in)), p)

layer1 = Dense(1024, activation='relu')
mask1 = K.dropout(K.ones((n_batch, n_layer)), p)

layer2 = Dense(1024, activation='relu')
mask2 = K.dropout(K.ones((n_batch, n_layer)), p)

layer3 = Dense(1024, activation='relu')
mask3 = K.dropout(K.ones((n_batch, n_layer)), p)

layer4 = Dense(1024, activation='relu')
mask4 = K.dropout(K.ones((n_batch, n_layer)), p)

softmax_layer = Dense(10, activation='softmax')

# apply model
out1 = layer1(dropout([x, maskx]))
out1 = layer2(mask1 * out1)
out1 = layer3(mask2 * out1)
out1 = layer4(mask3 * out1)
out1 = softmax_layer(mask4 * out1)

out2 = layer1((1. - maskx) * x)
out2 = layer2((1. - mask1) * out2)
out2 = layer3((1. - mask2) * out2)
out2 = layer4((1. - mask3) * out2)
out2 = softmax_layer((1. - mask4) * out2)

prediction = 0.5 * (out1 + out2)

# this creates a model that includes
# the Input layer and three Dense layers
model = Model(input=x, output=prediction)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=n_batch, nb_epoch=n_epoch,
                    verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

print(score)
