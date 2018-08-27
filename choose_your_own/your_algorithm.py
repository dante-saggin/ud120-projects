#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
learning_rate = 1
n_estimators = 10000
models = [ GaussianNB(),
            SVC(C=10000, kernel="rbf", gamma=90),
            RandomForestClassifier(n_estimators=30),
            tree.DecisionTreeClassifier(min_samples_split=18, max_depth=5),
            AdaBoostClassifier(SVC(C=10000, kernel="rbf", gamma=90),  learning_rate=learning_rate,  n_estimators=n_estimators, algorithm="SAMME")
            ]
acc_max = 0
for model in models:
    model.fit(features_train,labels_train)
    labels_predicted = model.predict(features_test)            
    acc = accuracy_score(labels_predicted, labels_test)
    if acc > acc_max :
    	acc_max = acc
    	clf = model
    	print  model
    	print "Accuracy " + str(acc)

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
