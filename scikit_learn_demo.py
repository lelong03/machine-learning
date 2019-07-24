from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# load data set
iris_dataset = load_iris()

# split data set into train set & test set
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# start training with decision tree classifier model
tree_model = DecisionTreeClassifier()
model = tree_model.fit(x_train, y_train)

# start testing after train
print(model.score(x_test, y_test))

# test with random data
x = np.array([[6.1, 2.8, 5.2, 1.7]])
print(model.predict(x))
