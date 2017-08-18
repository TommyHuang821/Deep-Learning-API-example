
# 如何直接從txt檔或是csv檔直接匯入資料到python workspace</br>
### 1. 最簡單方式是用numpy module來幫助匯入資料:</br>
    
    import numpy as np 
    
  如果你的檔案分割資料看是用什麼來分割</br>
  如果是逗號(",")分隔，則加入參數delimiter=','</br>
  如果是空白號(" ")分隔，則加入參數delimiter=' ' </br></br>
  如果匯入的資料是float型態，在dtype = float</br>
  如果匯入的分類的類別通常是整數型態，此時資料是int型態，在dtype = int</br>

*** 假設有個檔案是空白分隔的txt檔案(RegressionExample.txt)，可以直接用numpy內建的genfromtxtx來讀取資料，如下圖</br>

    data=np.genfromtxt("RegressionExample.txt", delimiter=' ',dtype =float)
    
  ![alt tag](https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/fig/txtdata_space.png "txt型態")</br>
  存出來的資料型態則是array，因為我是設成float，所以整數資料後面會有小數點帶.000</br>
  ![alt tag](https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/fig/txtdata_space_numpy.png)</br>


*** 假設有個檔案是csvt檔案(sampledata.csv)，則也是直接用numpy內建的genfromtxtx來讀取資料，如下圖</br>

    data=np.genfromtxt("sampledata.csv", delimiter=',',dtype =float)

  </br>![alt tag](https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/fig/csvdata.png)</br>      
  存出來的資料型態也是array</br>
  ![alt tag]( https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/fig/csvdata_numpy.png)</br>
 
 
 ### 2. 用open函數來幫助匯入資料</br>
  這種方法通常，是你對你的資料型態非常了解</br>
  因為它等於開檔案後，一行一行(row)讀取，讀進來的不是float是string</br>
  下例子，我們一樣去讀RegressionExample.txt這個檔案，裡面是</br>
  </br>192000 15 1800000 5800 50</br>
  190400 15 1790000 6200 50</br>
  ...</br></br>
  這樣的型態，所以for loop (i)則是先取出</br>
  </br>192000 15 1800000 5800 50</br></br>
  然後for loop (j)在針對 i="192000 15 1800000 5800 50 \n"這個字串處理</br>
  for j in i.split(' ')則是將這字串切割，遇到' '則分割，所以會將每個數字分割出來</br>
  </br>
  但最後的一個元素因為換行所以會是\n，所以要加入判斷式判斷是不是換行</br>
  此範例讀出來的資料會是list的型式，如果後續要做矩陣運算，也是用numpy.array轉換即可(record=np.array(record))。
  
    with open('D:\Tommy\ECG\Python\docs\RegressionExample.txt','r') as File:
        record =[[float(j) for j in i.split(' ') if j.strip('\n')!='0'] for i in File]

之後在介紹如果資料型態很複雜要怎麼處理。
