# K nearest neighbors - Classification

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("bank-full.csv", sep=";")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encode Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,2, 3,4,6,7,8,10,15])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Encode Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#split data into test and train in-order to train the model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = .25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set for k = 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors =13 , metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#calculate the accaracy 
from sklearn.metrics import accuracy_score
print('Accuracy:')
print(accuracy_score(y_test, y_pred))

from pandas import DataFrame
from sklearn.model_selection import cross_val_score

# 10-Fold Cross Validation and Averaged Accuracy
cv_score = cross_val_score(estimator= classifier, X =X, y = y ,cv= 10)
print((DataFrame(cv_score, columns= ['CV']).describe()).T)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)