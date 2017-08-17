import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt # 可视化模块

# create some data
X = np.linspace(-1, 1, 100)
np.random.shuffle(X)    # randomize the data
Y = 3 * X + 2 + np.random.normal(0, 0.05, (100, ))


X_train, Y_train = X[:60], Y[:60]     # train 前 160 data points
X_test, Y_test = X[60:], Y[60:]       # test 后 40 data points

 # build a neural network from the 1st layer to the last layer
model = Sequential()
model.add(Dense(output_dim=10, input_dim=1))
model.add(Dense(output_dim=1, input_dim=10, activation='linear'))
model.compile(loss='mse', optimizer='sgd') # choose loss function and optimizing method

print('Training -----------')
for step in range(1001):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 10 == 0:
        print('train cost: ', cost)
        
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

Y_pred = model.predict(X_test)
X_pred = model.predict(X_train)

# plot data
plt.subplot(211)
plt.scatter(X_train, Y_train)
plt.scatter(X_train, X_pred,color="r")
# plotting the prediction
plt.subplot(212)
plt.scatter(X_test, Y_test)
plt.scatter(X_test, Y_pred,color="r")
plt.show()