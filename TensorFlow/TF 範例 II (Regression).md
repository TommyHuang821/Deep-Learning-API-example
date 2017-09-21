# Python TensorFlow 實作神經網路範例 II:</br>
I. 用TensorFlow計算Linear Regression (梯度法)</br></br>
II. 用Closedform計算Linear Regression (唯一解)</br>
    Closedform推估請看: https://sites.google.com/site/personalpagechihshenghuang/classroom-news/iii-hui-gui-regression</br>
## [範例原始碼](https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/TensorFlow/Main_tf_Regression.py) 

1. Import 相關模組 (TensorFlow, Numpy, MatplotLib): </br>

       import tensorflow as tf
       import numpy as np
       import matplotlib.pyplot as plt

2. 產生模擬資料 (線性關係資料): </br>

       X = np.linspace(-1, 1, 100)[:,np.newaxis]
       np.random.shuffle(X)    # randomize the data
       Y =  3 * X + 2 + np.random.normal(0, 0.05, (100, 1)) # 線性

3.將產生資料分成訓練和測試資料集(前70個當訓練資料，後30個當測試資料): </br>
 
       X_train, Y_train = X[:70], Y[:70]     # train 前 60 data points
       X_test, Y_test = X[70:], Y[70:]          

4. 用TensorFlow框架出要推估的回歸模型:</br> 
   prediction = xs * w + b </br> 
   xs是自變數 </br>
   ys是應變數 </br>
   prediction是推估結果 </br>
   我們要找的參數是 w 和 b </br>
   目標函數(loss function):這邊用的是誤差最小平方和 </br>
   最佳解解法: 梯度法(Gradient Descent Optimizer)，學習率(Learning rate): 0.1 </br>
   
       dim_input=1;
       dim_output=1;
       ###########  Linear Regression by TensorFlow (GradientDescent) ###############
       xs = tf.placeholder(tf.float32,[None,dim_input],name='x_input')
       ys = tf.placeholder(tf.float32,[None,dim_output],name='y_output')
       
       w = tf.Variable(tf.random_normal([dim_input, dim_output]))
       b = tf.Variable(tf.zeros([1, dim_output]) + 1,)
       prediction = tf.matmul(xs, w) + b
       # loss function (prediction and real target)
       loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction-ys),
            reduction_indices=[1]),name='Loss')
       # optimal: find the Weights and biases
       train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
       sess = tf.Session()
       sess.run(tf.global_variables_initializer())


5. 執行TensorFlow找解: </br>
   這邊總學習次數上限為1000次，但當平方誤差根 (Root Mean Square Error, RMSE)不在變動時(收斂)則跳出迴圈。 </br>
   sess.run(train_step,feed_dict={xs: X_train, ys: Y_train}) 這邊在塞訓練資料進去學習模型。 </br>
   y_pre = sess.run(prediction, feed_dict={xs: X_test}) 這邊在計算訓練資料計算出相對應的預測結果。</br>
   rmse=np.append(mse,np.mean(np.sqrt(np.sum(np.square(Y_test-y_pre))))): RMSE計算。</br>
      
       rmse=np.array([])
       for i in range(1000):
           sess.run(train_step,feed_dict={xs: X_train, ys: Y_train})
           y_pre = sess.run(prediction, feed_dict={xs: X_test})
           rmse=np.append(mse,np.mean(np.sqrt(np.sum(np.square(Y_test-y_pre)))))
           if i>=1:
               if abs(rmse[i]-rmse[i-1])<np.spacing(1):
                   break
           print('Iternation' + str(i+1) + ': RMSE=' + str(mse[i]))
  
  計算結果:
  
      Iternation1: MSE=15.1094248271
      Iternation2: MSE=13.9553660398
      Iternation3: MSE=12.9156291043
      Iternation4: MSE=11.9732869555
      Iternation5: MSE=11.1148922075
      Iternation6: MSE=10.3296404788
      Iternation7: MSE=9.6087485555
        ...
      Iternation213: MSE=0.272796794685
      Iternation214: MSE=0.272796802641
      Iternation215: MSE=0.272796627189
      Iternation216: MSE=0.272796572243
        
5.2  用Closedform計算Linear Regression (唯一解): </br> 
      y = w * x + b  </br></br> 
      如果是多維度(multi-dimension)同樣適用此算法 </br> 
      Y = Beta * X  </br> 
      截距項(intercept)會被並到 X 裡面去。 </br> 
      ![alt tag](https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/fig/Regression_ClosedForm.png)</br>
      這邊我建了一個 function (Train_LSE_Regression)，裡面內容實際就是在算上述公式 </br>
      X_input_train=np.concatenate((X_train,np.ones([n_train,1])),axis=1) 這邊就是將截距項塞到資料內。</br> 
      原本X_train是一個70*1的矩陣，經由上述式子則X_input_train是70*2的矩陣，第一行是原本的資料(X_train)，第二行則是都是1的70*1矩陣。</br> 

      def Train_LSE_Regression(x,y):
          tmp1=np.matmul(np.transpose(x),x)
          tmp1=np.linalg.pinv(tmp1)
          tmp2=np.matmul(np.transpose(y),x)
          Weight=np.matmul(tmp1,np.transpose(tmp2))
          return Weight

      n_train=np.size(X_train,0)
      n_test=np.size(X_test,0)
      X_input_train=np.concatenate((X_train,np.ones([n_train,1])),axis=1)
      Weight_LSE=Train_LSE_Regression(X_input_train,Y_train)

      X_input_test=np.concatenate((X_test,np.ones([n_test,1])),axis=1)
      y_pre_LSE=np.matmul(X_input_test,Weight_LSE)
      mse_LSE=np.mean(np.sqrt(np.sum(np.square(Y_test-y_pre))))
      
      
      
6. 比較兩種方法計算出來的差異:

       print('TF (GD) Weights:' + str(sess.run(w)) + str(sess.run(b)))
       print('TF (GD) MSE:' + str(mse_tf))
       print('LSE     Weights:' + str(Weight_LSE[0]) + str(Weight_LSE[1]))
       print('LSE     MSE:' + str(mse_LSE))
           
   結果:
   
       TF (GD) Weights: [[ 3.00593758]][[ 2.00726533]]
       TF (GD) MSE: 0.272796572243
       LSE     Weights: [ 3.00593935][ 2.00726602]
       LSE     MSE: 0.272796572243
       
7.  結論:        
   從結果可以發現，梯度法執行到收斂後找到的答案跟唯一解找出來的答案基本上一致。</br> 
   雖然唯一解很快，只需要矩陣運算計算一組即可以得到答案</br> 
   但因為唯一解在高維度低樣本數時可能會有共變異數矩陣(covariance matrix)奇異(singular)的情況出現</br> 
   這時候可能會造成估計錯誤，造成不良的結果</br> 
   所以最好實際上在算的時候，還是用梯度法去執行Regression。
   
   
