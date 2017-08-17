# Python Keras 實作神經網路範例:
========================================================
### 用神經網路做回歸
*匯入numpy, Keras, 和 matplotlib模組 </br>
Numpy用來產生資料用，Keras神經網路Module, matplotlib用來畫圖

    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense
    import matplotlib.pyplot as plt
    
*利用線性模型產生100筆資料 (其中60筆當train，40筆當test)，主要是用numpy模組</br>



    X = np.linspace(-1, 1, 100)
    np.random.shuffle(X)    ## randomize the data

case I (Linear): y=3*x+2

    Y = 3 * X + 2 + np.random.normal(0, 0.05, (100, ))
    
case II (Square): y=3*x^2+2

    Y = 3 * X**2 + 2 + np.random.normal(0, 0.05, (100, ))
    
    X_train, Y_train = X[:60], Y[:60]     ## train 前 160 data points
    X_test, Y_test = X[60:], Y[60:]       ## test 后 40 data points

X軸是Input值(X_train)，Y軸是Output值(Y_train)，上面那張是training set，下面那張是testing set</br>
Linear Case (Case I): </br>
![alt tag](https://github.com/TommyHuang821/Note/blob/master/fig/RegressionCase_linear.png)</br>
Square Case (Case II)</br>
![alt tag](https://github.com/TommyHuang821/Note/blob/master/fig/RegressionCase_Square.png)</br>


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
    model.compile(loss='mse', optimizer='sgd') ## choose loss function and optimizing method

*網路訓練設定為1000次，每十次呈現一次train的cost的結果。</br>

    print('Training -----------')
    for step in range(1000):
    cost = model.train_on_batch(X_train, Y_train) ## 神經網路用training data的learning在此步驟
    if step % 10 == 0:
    print('train cost: ', cost)
            
*網路訓練完成後, 將測試資料帶入算cost的值</br>

    print('\nTesting ------------')
    cost = model.evaluate(X_test, Y_test)
    print('test cost:', cost)
    Y_pred = model.predict(X_test) 
    X_pred = model.predict(X_train)

*網路訓練完成後, 也可以將每一層的Weight的結果印出來看</br>  

    W, b = model.layers[0].get_weights()
    print('Weights=', W, '\nbiases=', b)



畫兩張結果圖</br>
第一張train data的真實資料(藍色)，預測出來得結果(紅色)</br>
第一張test data的真實資料(藍色)，預測出來得結果(紅色)</br>

    plt.subplot(211)
    plt.scatter(X_train, Y_train)
    plt.scatter(X_train, X_pred,color="r")
    plt.subplot(212)
    plt.scatter(X_test, Y_test)
    plt.scatter(X_test, Y_pred,color="r")
    plt.show()
 
# 測試結果:</br>
這邊我是兩種激活函數，第一種用Linear，第二種用relu。Hidden node都設100個，學習次數都設在10000次。</br>
*X軸是Input值(X_train)，Y軸是Output值(Y_train)，上面那張是training set，下面那張是testing set</br>
藍色的點是ground truth，紅色的點是預測出來的結果。</br>
### Linear Case (Case I):</br>
activation用Linear</br>
![alt tag](https://github.com/TommyHuang821/Note/blob/master/fig/RegressionCase_linear_result_linear_10000.png)</br>
activation用relu</br>
![alt tag](https://github.com/TommyHuang821/Note/blob/master/fig/RegressionCase_linear_result_relu_10000.png)</br>

### Square Case (Case II)</br>
activation用Linear</br>
![alt tag](https://github.com/TommyHuang821/Note/blob/master/fig/RegressionCase_Square_result_linear.png)</br>
activation用relu</br>
![alt tag](https://github.com/TommyHuang821/Note/blob/master/fig/RegressionCase_Square_result_relu_10000.png)</br>

# 結論:
這邊純粹先探討activation function，Hidden node個數不在討論範圍。</br>
從範例的結果得知</br>
當資料型態是線性(case I)的時候，在不論activation是線性(linear)或非線性(relu)，都可以神經網路都可以fit資料。</br>
但是當資料型態轉變為非線性情況(case II)，activation是線性(linear)就不能將資料fit的很好，在非線性(relu)才能得到好的結果。</br>
神經網路在第一個強項就在於非線性的activation，將資料從原始空間轉換到一個非線性的空間，讓模型可以更好去fit資料</br>
但當activation是線性時，資料在Hidden layer計算時，資料依舊都是在原始維度的空間轉換，資料不論怎麼轉換依舊是原始空間的分佈。</br>



