# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
`)Import the required packages.
2)Import the dataset to operate on.
3)Split the dataset.
4)Predict the required output.
5)End the program

## Program:
~~~
Program to implement the SVM For Spam Mail Detection..
Developed by: SYED MUHAMED ZAHI
RegisterNumber:  212221230114
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
result

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
~~~


## Output:
![image](https://user-images.githubusercontent.com/94187572/204822391-735e1ea6-f4be-4bc2-bf02-c292ec0c804a.png)
![image](https://user-images.githubusercontent.com/94187572/204822611-26c99893-ffce-4d56-b04f-e77474809a9f.png)
![image](https://user-images.githubusercontent.com/94187572/204822724-3b7eb376-6090-45f8-a4af-9caaf7218c41.png)

![image](https://user-images.githubusercontent.com/94187572/204822664-72a679a4-e549-4836-8a8e-2e4bdb606288.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
