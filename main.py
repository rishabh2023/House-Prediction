import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('datasets/house_data.csv')
# Matplotlib.pyplot
#plt.scatter(data['area'], data['price'])
#plt.show()
 
new_df = df.drop('price',axis='columns')
price = df.price
# Linear Regession
reg = linear_model.LinearRegression()
reg.fit(new_df, price)
pre = reg.predict([[6000]])
print(pre)
