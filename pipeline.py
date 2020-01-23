from scipy.spatial import distance

def getDist(a, b):
    return distance.euclidean(a, b)

class DIY_KNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        best_dist = getDist(row, self.X_train[0])
        best_ind = 0
        for i in range(1, len(self.X_train)):
            dist = getDist(row, self.X_train[i])
            if(dist < best_dist):
                best_dist = dist
                best_ind = i
        return self.y_train[best_ind]



# import dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.5)  # half of data used as training, half as testing

# # decision tree model
# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()


# k nearest neighbors model
# from sklearn.neighbors import KNeighborsClassifier
my_classifier = DIY_KNN()


my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)
# show accuracy of decision tree
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions))
