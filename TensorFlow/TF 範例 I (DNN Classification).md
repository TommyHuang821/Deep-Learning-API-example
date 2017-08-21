# Python TensorFlow 實作神經網路範例 I:
利用TensorFlow建立DNN進行Classification</br>

### [完整用TensorFlow做神經網路分類範例原始碼](https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/TensorFlow/Main_tf_DNN.py) 

*範例資料庫: 手寫數字辨識資料庫((MNIST))，AI大神Yann LeCun的資料庫連結: http://yann.lecun.com/exdb/mnist/</br>
  如果你沒有去下載資料庫也沒有差，TensoeFlow內有幫你載好。</br>

### 因為TensorFlow不同於Keras那種懶人包，使用TensorFlow必須要懂神經網路架構才有辦法建構出一個分類網路</br>
所以TensorFlow給初學者的教學範例課程，會有一些基本神經網路運作數學說明 (https://www.tensorflow.org/get_started/mnist/beginners)</br>
這此我也懶得打了，之後有空在做一份，MLP forwardfeed和backpropagation說明與推導。</br>

此份範例將可以學到</br>
* 用最笨的方法建立DNN structure，和用較為聰明一點的方式建立DNN structure</br>
* 不同optimier在學習上的優缺點(存粹看performance，原理請參照網路上找的最佳化理論課程)</br>
    
使用TensforFlow架構神經網路步驟  </br>
1. 匯入相關module 和匯入MNIST資料庫資料</br>

       import tensorflow as tf
       from tensorflow.examples.tutorials.mnist import input_data
       mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

   在TensorFlow內MNIST會先將資料分成train, test, validation</br>
   train顧名思義就是拿來訓練模型，validation則是拿來做模型的fine-tune，test則是用來驗證訓練好的模型的。</br>
   後面例子會稍微說明。</br>
       
       mnist.validation.num_examples
   Out[33]: 5000

       mnist.train.num_examples
   Out[34]: 55000

       mnist.test.num_examples
   Out[35]: 10000
  
  
2. 宣告一個輸入資料佔存器(placeholder)</br>
   因為此刻要開始建構模型，但輸入資料是什麼必須先宣告</br>
   xs為輸入的圖(會將28*28的圖轉成一個784的array) </br>
   ys為相對應的Label</br>
   label 是 0 則 ys=[1,0,0,0,0,0,0,0,0,0]</br>
   label 是 1 則 ys=[0,1,0,0,0,0,0,0,0,0]</br>
                ...</br>
   label 是 9 則 ys=[0,0,0,0,0,0,0,0,0,1]</br>

       xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
       ys = tf.placeholder(tf.float32, [None, 10])
      
      
3.  建立initial DNN structure </br>
    這邊我們要建立的網路架構如下</br>
    Input layer (784 nodes) → Hidden layer 1 (100 nodes) → Hidden layer 2 (50 nodes) → Output layer (10 nodes)</br>
    
    這邊我示範用最笨的方法去撰寫此神經網路</br>
    *input to hidden 1 
    
        Weights = tf.Variable(tf.random_normal([784, 100]))
        biases = tf.Variable(tf.zeros([1, 100]) + 0.1,)
        W_with_b = tf.matmul(xs, Weights) + biases
        h1_input = tf.nn.sigmoid(W_with_b)

    *hidden 1 to hidden 2 
    
        Weights = tf.Variable(tf.random_normal([100, 50]))
        biases = tf.Variable(tf.zeros([1, 50]) + 0.1,)
        W_with_b = tf.matmul(h1_input, Weights) + biases
        h2_input = tf.nn.sigmoid(W_with_b)

    *hidden 2 to output 
    
        Weights = tf.Variable(tf.random_normal([50, 10]))
        biases = tf.Variable(tf.zeros([1, 10]) + 0.1,)
        W_with_b = tf.matmul(h2_input, Weights) + biases
        prediction = tf.nn.softmax(W_with_b)
          
    大家有發現這一大段code架構起來的網路，其實都是在做一樣的事情嗎? </br>
    每個layer都是在產生一個weight matrix和biases，然後上一層的輸出和weight matrix相乘後加上biases</br>
    最後在看用什麼activation function做為輸出，差別指在input node的個數和input的資料。</br></br>
    
    所以比較聰明的做法是</br>
    直接define一個function去做這repeat的事情
    
        def tf_ModelAdd_layer(inputs, in_size, out_size, activation_function=None):
          Weights = tf.Variable(tf.random_normal([in_size, out_size]))
          biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
          W_with_b = tf.matmul(inputs, Weights) + biases
          if activation_function is None:
            outputs = W_with_b
          else:
            outputs = activation_function(W_with_b,)
          return outputs

    此範例是4層(3個connection layer)架構，假設今天是一個20層NN，用笨的方法code會寫到死</br>
    然後面上那一大串笨的方法可以減化成下列型式 (這邊是不是看起來跟Keras.model.add長得很像哩) </br>
    
        h1 = tf_ModelAdd_layer(xs, 784, 100,  activation_function=tf.nn.sigmoid)
        h2 = tf_ModelAdd_layer(h1, 100, 100,  activation_function=tf.nn.sigmoid)
        prediction = tf_ModelAdd_layer(h2, 100, 10,  activation_function=tf.nn.softmax)
   
   4. 宣告loss/cost function的型態 </br>
   網路建構好之後，不同於Keras那種懶人包，你必須要自己宣告你最後要讓什麼目標函數最小或是最大化，也就是loss/cost函數你想用什麼，你可以自己定義
   </br>在分類問題上，我們依舊用cross-entropy來當作loss/cost function (the error between prediction and real data)</br>
      
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    
   5. Optimizer </br>
   這邊是看你要用什麼最佳化方法來達到最佳解</br>
   此範例是希望loss/cost function越小越好，在此我採用兩種最佳化方法</br>
   
     第一Stochastic gradient descent(SGD)</br>
   
       train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

     第二個是Adam</br>
   
       train_step =tf.train.AdamOptimizer(learning_rate=0.1,beta1=0.9,beta2=0.99,epsilon=1e-8,name="Adam").minimize(cross_entropy)

   兩個選一種去執行即可</br>
   但因為Adam可以解決最佳解在鞍部或是local minimum等問題，而且因為有weight decay的效果，所以通常建議用
   Adam，同樣的架構較少的iternation次數即可以達到最佳解。</br>

   6. 執行</br>
   上述的程式除了import之外都是為了架構一個神經網路，在TensorFlow內架構好的網路，必須要有個類似啟動的方式將所有宣告的東西串起來</br>
   而且所有上面定義的東西都必須要先初始化，所以必須要執行下列程式啟動你架構好的TensorFlow NN。</br>

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(1000)
      sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
      if i % 50 == 0:
        acc = compute_accuracy(mnist.test.images, mnist.test.labels)
        print(acc)
   
   </br>
   然後這邊開始執行網路學習共1000次</br>
   mnist.train.next_batch(1000)是取1000個mini-batch出來做小批次的學習</br>
   batch_xs為輸入的1000筆資料的值(1000,784)</br>
   batch_ys為這1000筆資料對應的類別</br>

   sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})這邊程式讀法如下:</br>
   sess是執行train_step這個，train_step是跑Optimizer裡面要最小化cross_entropy</br>
   cross_entropy裡面要算ys和prediction</br>
   ys是你學習時資料的ground truth，prediction是架構好神經網路最後一層的輸出，計算來源為xs。</br>
   所以每一次學習指要放xs和ys這兩個東西進去train_step，他才能跑出一次完整的神經網路學習。</br>

   這邊供執行1000次，我這邊會放SGD和Adam在testing上的結果 </br>
   SGD:</br>
   ![alt tag](https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/fig/TF_SGD_demo.png)</br>
   Adam:</br>
   ![alt tag](https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/fig/TF_ADAM_demo.png)</br>
   
   由上圖可知</br>
   *Adam:在train的資料的正確率是0.96083635跟test資料的結果0.9436不會差太多。</br>
   
   *SGD在train的資料的正確率是0.76907271，在test資料的結果是0.76749998。</br>
   在train和test一直都不是很理想，所以可以排除overfitting的問題</br>
   比較有有可能問題是掉到local minimum內，因為網路沒有很複雜，所以可能有掉到local minimum，網路結構深又大，掉到local minimum機率會變小</br>
   
   所以我把SGD learning rate調高到0.5 (其實Tensorflow的範例設定是0.5，但我為了呈現不同最佳化方法的好處，所以稍微修改一下)</br>
   SGD(learning rate=0.5):</br>
   ![alt tag](https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/fig/TF_SGD_demo2.png)</br>

   結果SGD(LR=0.5)有比SGD(LR=0.1)來的好，但還是沒有Adam這麼快到達最佳的結果。</br>
   下圖是用來呈現不同最佳化方法的收斂速度，與他是否能跳出Local minimum</br>
   ![alt tag](http://ruder.io/content/images/2016/09/contours_evaluation_optimizers.gif)</br>
   ![alt tag](http://ruder.io/content/images/2016/09/saddle_point_evaluation_optimizers.gif)</br>
   因為上兩張圖我是直接複製別人的連結，所以沒有Adam在裡面比較，但因為Adam是RMSProp與Momentum兩個方法的合成，所以會比較好。</br>
   有興趣看最佳化東西的人，請參考:http://ruder.io/optimizing-gradient-descent/ </br>      
       

   最後附上計算準確率的function。</br>  
   
      def compute_accuracy(v_xs, v_ys):
        global prediction
        y_pre = sess.run(prediction, feed_dict={xs: v_xs})
        correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
        return result
