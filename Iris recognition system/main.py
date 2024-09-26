import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

iris=pd.read_csv("iris.csv")
print(iris)
print(iris.shape)
print(iris.describe())

# Check for null values
print(iris.isna().sum())
print(iris.describe())

print(iris.head())
print(iris.head(150))
print(iris.tail(100))

# Checking for outliers
import matplotlib.pyplot as plt
plt.figure(1)
plt.boxplot([iris['Sepal.Length']])
plt.figure(2)
plt.boxplot([iris['Sepal.Width']])
plt.show()
iris.hist()
plt.show()

X = iris['Sepal.Length'].values.reshape(-1,1)
print(X)

Y = iris['Sepal.Width'].values.reshape(-1,1)
print(Y)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.scatter(X, Y, color='b')
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
train, test = train_test_split(iris, test_size = 0.25)
print(train.shape)
print(test.shape)
train_X = train[['Sepal.Length', 'Sepal.Width', 'Petal.Length',
                 'Petal.Width']]
train_y = train.Species

test_X = test[['Sepal.Length', 'Sepal.Width', 'Petal.Length',
                 'Petal.Width']]
test_y = test.Species

train_X.head()

test_y.head()
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('Accuracy:',metrics.accuracy_score(prediction,test_y))
#Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(test_y,prediction)
print("Confusion matrix: \n",confusion_mat)