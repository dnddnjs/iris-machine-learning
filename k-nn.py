"""
code : https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset/notebook
"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('C:/Users/dnddn/PycharmProjects/iris-machine-learning'
                   '/data/Iris.csv')

X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']
print("input data shape is ", X.shape)
print("target data shape is ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=5)
# experimenting with different n values
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()

knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X, y)

# make a prediction for an example of an out-of-sample observation
pred = knn.predict([[6, 3, 4, 2]])
print(pred)