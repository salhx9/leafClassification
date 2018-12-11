import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def encode(train, test):
    le = LabelEncoder().fit(train.species) 
    labels = le.transform(train.species)           # encode species strings
    classes = list(le.classes_)                    # save column names for submission
    test_ids = test.id                             # save test ids for submission
    
    train = train.drop(['species', 'id'], axis=1)  
    test = test.drop(['id'], axis=1)
    
    return train, labels, test, test_ids, classes

train, labels, test, test_ids, classes = encode(train, test)
train.head(1)

sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

for train_index, test_index in sss:
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]


classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(),
    ExtraTreesClassifier(),
    RandomForestClassifier(),
    LogisticRegression()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

# this thing yields the log loss for each classifier method above
for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)

# need to fine tune this for our dataset 

# randomforest classifier
favorite_clf0 = RandomForestClassifier()
favorite_clf0.fit(X_train, y_train)
test_predictions0 = favorite_clf0.predict_proba(test)
# Format DataFrame
submission0 = pd.DataFrame(test_predictions0, columns=classes)
submission0.insert(0, 'id', test_ids)
submission0.reset_index()
# Export Submission
submission0.to_csv('submissionRFC.csv', index = False)
submission0.tail()

#extra trees classifier
favorite_clf1 = ExtraTreesClassifier()
favorite_clf1.fit(X_train, y_train)
test_predictions1 = favorite_clf1.predict_proba(test)
# Format DataFrame
submission1 = pd.DataFrame(test_predictions1, columns=classes)
submission1.insert(0, 'id', test_ids)
submission1.reset_index()
# Export Submission
submission1.to_csv('submissionETC.csv', index = False)
submission1.tail()


#linear discriminant analysis
favorite_clf2 = LogisticRegression()
favorite_clf2.fit(X_train, y_train)
test_predictions2 = favorite_clf2.predict_proba(test)
# Format DataFrame
submission2 = pd.DataFrame(test_predictions2, columns=classes)
submission2.insert(0, 'id', test_ids)
submission2.reset_index()
# Export Submission
submission2.to_csv('submissionLR.csv', index = False)
submission2.tail()

# kNN
favorite_clf3 = KNeighborsClassifier(3)
favorite_clf3.fit(X_train, y_train)
test_predictions3= favorite_clf3.predict_proba(test)
# Format DataFrame
submission3 = pd.DataFrame(test_predictions3, columns=classes)
submission3.insert(0, 'id', test_ids)
submission3.reset_index()
# Export Submission
submission3.to_csv('submissionKNN.csv', index = False)
submission3.tail()

#decision tree classifier
favorite_clf4 = DecisionTreeClassifier()
favorite_clf4.fit(X_train, y_train)
test_predictions4= favorite_clf4.predict_proba(test)
# Format DataFrame
submission4 = pd.DataFrame(test_predictions4, columns=classes)
submission4.insert(0, 'id', test_ids)
submission4.reset_index()
# Export Submission
submission4.to_csv('submissionDTC.csv', index = False)
submission4.tail()



#https://www.datacamp.com/community/tutorials/random-forests-classifier-python