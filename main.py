import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#reading data file
data = pd.read_csv('glass.csv')

#spliting data as X and Y
x = data.iloc[:, :-1].values
y = data.iloc[:, [-1]].values


y = np.ravel(y)
X_sparse = coo_matrix(x)

#shuffling data
x, X_sparse, y = shuffle(x, X_sparse, y, random_state=0)

#detecting test and train set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


#selecting K value
classifier = KNeighborsClassifier(n_neighbors=3)


classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

