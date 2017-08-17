# Python Keras 實作神經網路範例 II:
### 利用Keras建立神經網路進行Classification</br>
* 範例資料庫: 手寫數字辨識資料庫((MNIST))，AI大神Yann LeCun的資料庫連結: http://yann.lecun.com/exdb/mnist/</br>
  如果你沒有去下載資料庫也沒有差，Keras裡面有幫你載好。</br>

基本上所有用TensorFlow或是其他module的課程都是用此(MNIST)資料庫當作範例來執行。</br>
此手寫數字資料庫內有幫你分類好60,000筆訓練資料(train set)，和10,000筆測試資料(test set)。</br>

每張圖的size是28*28(像數) 8 bits，下圖是從網路找別人將數字呈現出來。</br>
![alt tag](http://simonwinder.com/wp-content/uploads/2015/07/mnistExamples.png)</br>
   
1.和Keras範例I一樣，先將需要的module匯入</br>

    import numpy as np
    from keras.datasets import mnist
    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.optimizers import RMSprop
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

下載MNIST的資料會被存在'~/.keras/datasets/'，如果你第一次跑這個code又沒有資料庫，他會幫你下載。</br>
X_train shape (60,000, 28,28), y_train shape (60,000, )</br>
X_test shape (10,000, 28,28), y_test shape (10,000, )</br>
y是對應的ground truth </br>

2.資料前處理</br>
因為我們是要跑一般的神經網路，所以需要將每張圖的pixel資料(1,28,28)拉成一個array (1,28*28)=(1,784)</br>
然後8bits資訊(0-255)除255，將資料正規化到(0-1)</br>
因為是分類問題，所以output不會像regression只有ㄧ個輸出值，因為有10類(數字0~數字9)所以會有10個輸出值(可以當作每一張圖判給每一類的機率)。</br>
y_train經由下面的code會轉換成(60,000,10)，y_test會變成(10,000,10)</br>

    X_train = X_train.reshape(X_train.shape[0], -1) / 255.  
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.     
    y_train = np_utils.to_categorical(y_train, nb_classes=10)
    y_test = np_utils.to_categorical(y_test, nb_classes=10)

3.神經網路建立
網路架構是input(784 nodes)→Hidden1 (32 nodes)→Output(10 nodes)</br>
這邊的Keras神經網路架構，會和範例I不同，是Keras的另一種寫法。</br>
大家可以試試看，這種應該是比較多人在寫的方式。</br>
最佳化方法改成RMSprop來測試。</br>
分類的問題loss function一般都是用cross-entropy的方式來達到。</br>

    model = Sequential([
        Dense(32, input_dim=784), Activation('relu'),
        Dense(10), Activation('softmax'),
    ])
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(optimizer=rmsprop,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
4.參數和模型都建立後，可以用'model.fit'來學習模型，此範例mini-batch為500，epoch為10次</br>

    model.fit(X_train, y_train, nb_epoch=10, batch_size=500)

5.Evaluation</br>

    loss, accuracy = model.evaluate(X_test, y_test)
    print('\nTesting performance: \n')
    print('test loss: ', loss)
    print('test accuracy: ', accuracy)

下列模型我epoch都是設為10，結果比較如下:
此模型input(784 nodes)→Hidden1 (32 nodes)→Output(10 nodes)</br>
test loss:  0.154928062918</br>
test accuracy:  0.9538</br>

模型改成input(784 nodes)→Hidden1 (100 nodes)→Output(10 nodes)</br>
test loss:  0.103201441364</br>
test accuracy:  0.9694</br>

模型改成input(784 nodes)→Hidden1(100 nodes)→Hidden2(100 nodes)→Output(10 nodes)</br>
test loss:  0.0879571246014</br>
test accuracy:  0.9739</br>

### Hint:
  其實MNIST資料庫在NN下，準確度已經很高，如果epoch設多一點(讓NN多學一些)，也許結構不用太大就可以得到非常好的結果(除非overfitting問題產生)。</br>
  我用模型input(784 nodes)→Hidden1 (100 nodes)→Output(10 nodes) 執行50次</br>
  test loss:  0.0991017446712</br>
  test accuracy:  0.9775</br>
  結果跟很複雜的模型跑10次差不多，但訓練時間可以減少很多，這些問題在GPU下其實都解決了，但我沒有GPU只能用CPU版本去learn NN。</br>
