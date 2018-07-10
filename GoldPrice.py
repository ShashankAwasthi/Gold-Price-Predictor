import pandas as pd
import quandl, math, datetime 
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from sklearn.linear_model import LinearRegression

style.use('ggplot')

Df = quandl.get('NSE/SBIGETS')
Df=Df[['Close']] 

Df= Df.dropna()
Df.Close.plot(figsize=(10,5)) 

plt.ylabel("Gold ETF Prices")

plt.show()
Df['S_3'] = Df['Close'].shift(1).rolling(window=3).mean() 

Df['S_9']= Df['Close'].shift(1).rolling(window=9).mean() 

Df= Df.dropna() 

X = Df[['S_3','S_9']] 

X.head()

print(Df.head())
y = Df['Close']

y.head()
print(Df.head())

t=.8 
t = int(t*len(Df))

X_train = X[:t] 
y_train = y[:t]

X_test = X[t:] 
y_test = y[t:]

m1 = 1.15
m2 = 0.18
C = 0.39
X1 = Df['S_3']
X2 = Df['S_9']
Y = m1 * X1 + m2 * X2 + C
print(Y)
linear = LinearRegression().fit(X_train,y_train) 

print("Gold ETF Price =", round(linear.coef_[0],2),"* 3 Days Moving Average", round(linear.coef_[1],2),"* 9 Days Moving Average +", round(linear.intercept_,2))

predicted_price = linear.predict(X_test)  

predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])  

predicted_price.plot(figsize=(10,5))  

y_test.plot()  

plt.legend(['predicted_price','actual_price'])  

plt.ylabel("Gold ETF Price")  

plt.show()
r2_score = linear.score(X[t:],y[t:])*100  

float("{0:.2f}".format(r2_score))
print(r2_score)

















