
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import pylab as pl
from sklearn.svm import SVC
import sklearn.metrics as metrics

data = pd.read_csv("path where the data is saved.csv")
target = {"good":0, "bad":1}
data['class'] = data['class'].map(target)

rev_data = pd.get_dummies(data)
cols_at_end = ['class']
df = rev_data[[c for c in rev_data if c not in cols_at_end] + [c for c in cols_at_end if c in rev_data]]
df.to_csv("/Users/rincygeorge/Desktop/DS/Assign 3/Credit_Data_New2.csv")

data= pd.read_csv("path where the data is saved.csv")
cols = ['class']
df = data[[c for c in data if c not in cols] + [c for c in cols if c in data]]

train = df.iloc[:,:-1]
target = df.iloc[:,-1:]

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.4,stratify=target, random_state=3)
s=preprocessing.StandardScaler()
X_train=s.fit_transform(X_train)
X_test=s.transform(X_test)

#Logistic Regression
print("Logistic Regression:\n")
logreg = LogisticRegression().fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

y_pred = logreg.predict(X_test)
print("The accuracy score for the Logistic regression classifier is: ")
print(metrics.accuracy_score(y_test, y_pred))
print("Given below is the classification report for the Logistic regression classifier: ") 
print(metrics.classification_report(y_test, y_pred, target_names=['Good', 'Bad']))

probs = logreg.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, probs[:,1])
area = auc(recall, precision)
print("Area Under Curve: %0.2f" % area)

#Linear SVC
print("\n\nLinear SVC:\n")
clf = SVC(kernel='linear', probability=True, random_state=33).fit(X_train, y_train)
print("SVC Regression: Training set score: {:.3f}\n".format(clf.score(X_train, y_train)))
print("SVC Regression: Test set score: {:.3f}\n".format(clf.score(X_test, y_test)))

y_pred = clf.predict(X_test)
print("The accuracy score for the Linear SVC classifier is: %.2f"  %metrics.accuracy_score(y_test, y_pred))
print("Given below is the classification report for the Linear SVC classifier: ") 
print(metrics.classification_report(y_test, y_pred, target_names=['GOOD', 'BAD']))

probs = clf.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, probs[:,1])
area = auc(recall, precision)
print("Area Under Curve: %0.2f" % area)


#Decision tree
print("\n\nDecision Tree:\n")
tree = DecisionTreeClassifier(max_depth=4, criterion='entropy', random_state=3)
dtree=tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(dtree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(dtree.score(X_test, y_test)))
y_pred = dtree.predict(X_test)
print("The accuracy score for the decision tree classifier is: %.2f"  %metrics.accuracy_score(y_test, y_pred))
print("Feature importances:\n{}".format(dtree.feature_importances_))


y_pred = dtree.predict(X_test)
print("The accuracy score for the Decision tree classifier is: ")
print(metrics.accuracy_score(y_test, y_pred))
print("Given below is the classification report for the Decision tree classifier: ") 
print(metrics.classification_report(y_test, y_pred, target_names=['Good', 'Bad']))

probs = dtree.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, probs[:,1])
area = auc(recall, precision)
print("Area Under Curve: %0.2f" % area)


#Random forest
print("\n\n Random Forest:\n")
forest = RandomForestClassifier(n_estimators=100,max_depth=3, random_state=3)
rtree=forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rtree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rtree.score(X_test, y_test)))
y_pred = rtree.predict(X_test)
print("The accuracy score for the random forest classifier is: %.2f"  %metrics.accuracy_score(y_test, y_pred))

print("Given below is the classification report for the Random forest classifier: ") 
print(metrics.classification_report(y_test, y_pred, target_names=['Good', 'Bad']))

probs = rtree.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, probs[:,1])
area = auc(recall, precision)
print("Area Under Curve: %0.2f" % area)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
top_k = 5
new_indices = indices[:top_k]
# Print the feature ranking
print("Feature ranking:")
for f in range(top_k):
    print("%d. feature %d %s (%f)" % (f + 1, new_indices[f], train.columns[f], importances[new_indices[f]]))
#Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(top_k), importances[new_indices], color="r", align="center")
plt.xticks(range(top_k), new_indices)
plt.xlim([-1, top_k])
plt.show()



#Multi layered perceptron
print("\n\nMulti layered perceptron:\n")
mlp = MLPClassifier(max_iter=5000, alpha=10,random_state=3) 
mlpc=mlp.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(mlpc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlpc.score(X_test, y_test)))

y_pred = mlpc.predict(X_test)
print("The accuracy score for Multi layered perceptron classifier is: %.2f" %metrics.accuracy_score(y_test, y_pred))
print("Given below is the classification report for Multi layered perceptron classifier: ") 
print(metrics.classification_report(y_test, y_pred, target_names=['Good', 'Bad']))

probs = mlpc.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, probs[:,1])
area = auc(recall, precision)
print("Area Under Curve: %0.2f" % area)