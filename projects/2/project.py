from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as sklearn_DTF

from Tree import *

if __name__ == "__main__":

    # loading dataset
    iris = load_iris()
    
    #defining training and testing set
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    
    # Create and train the classifiers
    my_clf = DecisionTreeClassifier()
    my_clf.fit(X_train, y_train)
    
    sklearn_clf = sklearn_DTF()
    sklearn_clf.fit(X_train, y_train)

    # Make predictions
    my_y_pred = my_clf.predict(X_test)
    sklearn_y_pred = sklearn_clf.predict(X_test)

    # Calculate accuracies
    my_accuracy = np.sum(my_y_pred == y_test) / len(y_test)
    sklearn_accuracy = np.sum(sklearn_y_pred == y_test) / len(y_test)

    print(f"Accuracy of my model: {my_accuracy}\nAccuracy of sklearn: {sklearn_accuracy}")