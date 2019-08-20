import numpy as np 
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import cross_validate

#SVM Padrãozão
clf = svm.SVC(gamma='scale')
#Lendo dados
data = pd.read_csv("radiomics.csv", usecols= lambda column : column not in ["id","malignancy","class"])
n_samples = len(data.index)
#Normalizando
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = min_max_scaler.fit_transform(data)
#Capturando as classes
classes = pd.read_csv("radiomics.csv", usecols=["class"])
classes = classes.values.ravel()

#Trainando com metade
clf.fit(data_scaled[n_samples // 2:],classes[n_samples // 2:])
## Teste usando 50/50
expected = classes[:n_samples // 2]
predicted =  clf.predict(data_scaled[:n_samples // 2])
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))

## 10-fold
scores = cross_validate(clf, data_scaled,classes, scoring='roc_auc',cv=10)
avg_scores = np.average(scores['test_score'])
print(avg_scores)