import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib import pyplot
ax = plt.gca()

dataset = pd.read_csv("C:/Users/flash/Desktop/ML/project2/kc_house_data.csv/kc_house_data.csv")
price = dataset['price']/100
sqft_above = dataset['sqft_above']/100
def gradient_descent(x,y):
    m = 0
    b = 0
    
    n = len(x)
    learning_rate = 0.0001
    for i in range(1000):
        ypredicted = m * x + b
        #cost = (1.0/n)*sum([val**2 for val in (y-ypredicted)])
        md = -(2.0/n)*sum(x*(y-ypredicted))
        bd = -(2.0/n)*sum(y-ypredicted)
        m = m-(md*learning_rate)
        b = b-(bd*learning_rate)
    return m,b

m0,b0 = gradient_descent(sqft_above,price)
x0 = np.arange(0,100,1)
y0 = m0*x0+b0

pyplot.scatter(sqft_above,price)
plt.plot(x0,y0)
pyplot.xlabel("sqft_above/100")
pyplot.ylabel("house price")
pyplot.title("learningrate = 0.0001")
pyplot.show()
        
