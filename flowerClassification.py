# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)


# # box and whisker plots
# dataset.plot(kind = 'box', subplots = True, layout = (2, 2), sharex = False, sharey = False)
# pyplot.show()

# # histogram plots   
# dataset.hist()
# pyplot.show()

# # scatter plot matrix
# scatter_matrix(dataset)
# pyplot.show()

# split validation dataset
array = dataset.values
x = array[:, 0:4]
y = array[:, 4]
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

print(x.shape)
print(y.shape)
# # test different algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr'))) # logistic regression
# models.append(('LDA', LinearDiscriminantAnalysis()))                             # linear discriminant analysis
# models.append(('KNN', KNeighborsClassifier()))                                   # k-nearest neighbours
# models.append(('CART', DecisionTreeClassifier()))                                # classification and regression trees
# models.append(('NB', GaussianNB()))                                              # gaussian naive bayes
# models.append(('SVM', SVC(gamma='auto')))                                        # support vector machines

# # eval different algorithms
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' %(name, cv_results.mean(), cv_results.std()))
 

# # plot algorithm accuracy
# pyplot.boxplot(results, labels=names)
# pyplot.show()

# make predictions on validation dataset
model = SVC(gamma='auto') # support vector machines
model.fit(x_train, y_train)
predictions = model.predict(x_validation)

# evaluate predictions
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))