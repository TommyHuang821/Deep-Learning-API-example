# Python Keras 實作神經網路範例:
========================================================
### 用神經網路做回歸
*匯入numpy, Keras, 和 matplotlib模組 </br>
Numpy用來產生資料用，Keras神經網路Module, matplotlib用來畫圖

    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense
    import matplotlib.pyplot as plt
    
*利用線性模型產生100筆資料 (其中60筆當train，40筆當test)</br>
主要是用numpy模組</br>
    X = np.linspace(-1, 1, 100)
    np.random.shuffle(X)    # randomize the data
    Y = 3 * X + 2 + np.random.normal(0, 0.05, (100, ))
    X_train, Y_train = X[:60], Y[:60]     # train 前 160 data points
    X_test, Y_test = X[60:], Y[60:]       # test 后 40 data points

*建立神經網路結構和參數</br>
此範例為 神經網路架構為 Input→Hidden1→Output</br>
X為一個維度輸入資料(input)，Y為反應變數(Output)</br>
Hidden: 設 10 node且激活函數(activation function)為tanh</br>
其他 activation function 請見: https://keras-cn.readthedocs.io/en/latest/other/activations/</br>
-神經網路目標函數(loss/cost function)設定為mean square error(mse)，找參數最佳化的方法設定為Stochastic gradient descent optimizer(sgd)</br>
其他參數細節設定參閱: https://keras.io/losses/,  https://keras.io/optimizers/</br>

    model = Sequential()
    model.add(Dense(output_dim=10, input_dim=1, activation='tanh'))  ## (input→Hidden1)
    model.add(Dense(output_dim=1, input_dim=10)) ## (Hidden1→Output)
    model.compile(loss='mse', optimizer='sgd') # choose loss function and optimizing method

*網路訓練設定為1000次，每十次呈現一次train的cost的結果。</br>

    print('Training -----------')
    for step in range(1000):
        cost = model.train_on_batch(X_train, Y_train) # 神經網路用training data的learning在此步驟
        if step % 10 == 0:
            print('train cost: ', cost)
            
*網路訓練完成後, 將測試資料帶入算cost的值</br>
    print('\nTesting ------------')
    cost = model.evaluate(X_test, Y_test)
    print('test cost:', cost)
    Y_pred = model.predict(X_test) # 算Test data從NN模型估計出來結果
    X_pred = model.predict(X_train) # 算Train data從NN模型估計出來結果

*網路訓練完成後, 也可以將每一層的Weight的結果印出來看</br>    
    W, b = model.layers[0].get_weights()
    print('Weights=', W, '\nbiases=', b)



# plot data
* 畫兩張圖
第一張train data的真實資料(藍色)，預測出來得結果(紅色)
第一張test data的真實資料(藍色)，預測出來得結果(紅色)
    plt.subplot(211)
    plt.scatter(X_train, Y_train)
    plt.scatter(X_train, X_pred,color="r")
    plt.subplot(212)
    plt.scatter(X_test, Y_test)
    plt.scatter(X_test, Y_pred,color="r")
    plt.show()
