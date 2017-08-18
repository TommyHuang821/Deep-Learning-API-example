import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)

print(X_train[1].shape)

# input(784 nodes)→Hidden1 (32 nodes)→Output(10 nodes)
model = Sequential([
    Dense(100, input_dim=784), Activation('relu'),
    Dense(10), Activation('softmax'),
])
# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=50, batch_size=500)
loss, accuracy = model.evaluate(X_test, y_test)

print('\nTesting performance: \n')
print('test loss: ', loss)
print('test accuracy: ', accuracy)
