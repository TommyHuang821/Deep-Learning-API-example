<h1> Python Keras 實作神經網路範例:</h1>
========================================================
<h2> 用神經網路做回歸 </h1>
匯入numpy, Keras, and matplotlib模組
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense
    import matplotlib.pyplot as plt # 畫圖用



### 利用線性模型產生100筆資料 (其中60筆當train，40筆當test)
X = np.linspace(-1, 1, 100)
np.random.shuffle(X)    # randomize the data
Y = 3 * X + 2 + np.random.normal(0, 0.05, (100, ))
X_train, Y_train = X[:60], Y[:60]     # train 前 160 data points
X_test, Y_test = X[60:], Y[60:]       # test 后 40 data points

### 建立神經網路結構和參數 此範例為(Input→Hidden1→Output)
model = Sequential()
model.add(Dense(output_dim=10, input_dim=1, activation='tanh'))  ## (input→Hidden1, 在Hidden 1的activation函數是tanh)
model.add(Dense(output_dim=1, input_dim=10)) ## (Hidden1→Output)
model.compile(loss='mse', optimizer='sgd') # choose loss function and optimizing method

### 呈現training的cost的結果，每十次呈現一次，共跑1000次。
print('Training -----------')
for step in range(1000):
    cost = model.train_on_batch(X_train, Y_train) # 神經網路用training data的learning在此步驟
    if step % 10 == 0:
        print('train cost: ', cost)

### 
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
