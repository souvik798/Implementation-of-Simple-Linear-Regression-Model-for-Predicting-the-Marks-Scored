# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: K.Balaji
RegisterNumber: 212221230011
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='black')
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE= ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE= ',mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)

```

## Output:
### df.head():
![ou](./head.png)
### df.tail():
![ou](./tail.png)
### Array value of X:
![ou](./X.png)
<br>
<br>
### Array value of Y:
![ou](./Y.png)
### Values of Y prediction:
![ou](./py.png)
### Array values of Y test:
![ou](./yt.png)
### Training Set Graph:
![ou](./gt.png)
### Test Set Graph:
![ou](./gte.png)
### Values of MSE, MAE and RMSE:
![ou](./last.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
