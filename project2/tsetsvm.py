import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from SVM import SVM
ax = plt.subplot()

column_labels = ["id",  "radius", "texture","perimeter",
                                "area", "smoothness","compactness",
                                "concavity", "concave points", "symmetry", "fractal dimension"]
dataset = pd.read_csv("C:/Users/flash/Desktop/ML/project2/breast-cancer-wisconsin.data", names = column_labels)

dataset = dataset.replace(to_replace='?', value = np.nan)
dataset = dataset .dropna(how = 'any')

dataset2 = dataset.drop(['id'], axis=1)
dataset2['fractal dimension'] = dataset2['fractal dimension'].map({4:1, 2:0})

features = dataset2.drop('fractal dimension', axis=1)
labels = dataset2['fractal dimension']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

svmclassifier = SVC()
svmclassifier.fit(X_train, y_train)

x_array = np.array(X_train, dtype=int)
y_array = np.array(y_train, dtype=int)
pos = x_array[np.where(y_array==1)]
neg = x_array[np.where(y_array==0)]

#pyplot.scatter(pos[:,8],neg[:,8], c='r', label='Malignant')
ax.scatter(pos[:,5],pos[:,1], c='r', label='Malignant')
ax.scatter(neg[:,5],neg[:,1], c='b', label='Benign')
pyplot.show()