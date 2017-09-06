# TensorFlow Python模組安裝
會進來看這頁的基本上應該知道TensorFlow是什麼，所以介紹就免了</br>

## 直接開始安裝步驟</br>

以下會分成兩種安裝方式和驗證方式介紹:</br>
I.	如果Python環境設定都有完成的話，基本上就直接用pip install tensorflow就可以直接取安裝tensorflow模組
II.	如果無法用第一種方法正確安裝，可以參考別種作法
III. 如何驗證TensorFlow是否正確安裝

## I.	如果Python環境設定都有完成的話，基本上就直接用pip install tensorflow就可以直接取安裝tensorflow模組</br>
步驟如下: </br>
1. 打開執行命令，Python所有的模組安裝或是更新都要用此步驟，前一章安裝Anaconda有提。</br>
2. 執行命令視窗，直接打指令，如果你有GPU 則是裝GPU版的tensorflow</br>

       pip install tensorflow
       
       或
       
       pip install tensorflow-gpu
   
  我的電腦沒有GPU，所以此次範例以CPU版本為主。</br>
  ![alt tag](https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/fig/Tensorflow%E5%AE%89%E8%A3%9D1.png)</br>

3. 成功安裝則會有下面畫面，要看到Successfully installed tensorflow-版本編號，此次範例版本為1.3.0。</br>
  ![alt tag](https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/fig/Tensorflow%E5%AE%89%E8%A3%9D2.png)</br>

## II.	如果無法用第一種方法正確安裝，可以參考下列別種作法</br>
參考網頁: https://stackoverflow.com/questions/43419795/how-to-install-tensorflow-on-anaconda-python-3-6</br>
我之前有另一台電腦是用此方式安裝，可以成功安裝，在此我就只是純粹翻譯成中文。</br></br>
步驟如下: </br>
1.	到https://www.continuum.io/downloads 下載Anaconda Python 3.6 version for Window 64bit. (如果有安裝好Anaconda可以跳過)。</br>
2.	到執行命令下，建立一個名叫tensorflow的conda環境:</br>

        C:\> conda create -n tensorflow
       
3.	激活這個conda環境</br>

        C:\> activate tensorflow (tensorflow)
  
4.	到 http://www.lfd.uci.edu/~gohlke/pythonlibs/  下載模組</br>
(下載後的檔案路徑要記得後面會用到，此人的路徑是C:\Users\Joshua\Downloads\)</br>
2017/09/06時tensorflow模組有</br>
       tensorflow 1.3.0 cp35 cp35m win_amd64.whl</br>
       tensorflow 1.3.0 cp36 cp36m win_amd64.whl</br>
       tensorflow_gpu 1.1.0 cp35 cp35m win_amd64.whl</br>
       tensorflow_gpu 1.1.0 cp36 cp36m win_amd64.whl</br>


5.	安裝TensorFlow模組(路徑要記得修改成檔案剛剛下載的位置，底線部分要改成你剛剛下載的TensorFlow): </br>

       (tensorflow)C:>pip install C:\Users\Joshua\Downloads\ tensorflow-1.0.1-cp36-cp36m-win_amd64.whl
  
  ![alt tag](https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/fig/Tensorflow%E5%AE%89%E8%A3%9D5.png)</br>

6.	驗證剛剛安裝好的TensorFlow。

## III. 如何驗證TensorFlow是否正確安裝
到你自己python環境測試tensorflow，簡單印出Hello, TensorFlow!。 </br>
             
       import tensorflow as tf
       hello = tf.constant('Hello, TensorFlow!')
       sess = tf.Session() 
       print(sess.run(hello))
 
1. 在Spyder下執行:</br>
  ![alt tag](https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/fig/Tensorflow%E5%AE%89%E8%A3%9D3.png)</br>


2. 在執行命令下開始ipython執行:</br>
  ![alt tag](https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/fig/Tensorflow%E5%AE%89%E8%A3%9D4.png)</br>


此份講義就到此，有沒有覺得安裝TensorFlow很簡單啊</br>
