from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from utils import *

# loading dataset
iris = load_iris()['data']

#defining training and testing set
training_set, testing_set = split_array(iris, 0.8)

print(training_set, testing_set)

# Split dataset into training set and test set
#X_train, X_test, y_train, y_test = train_test_split(iris, test_size=0.2, random_state=1)
#print(X_train, X_test, y_train, y_test)