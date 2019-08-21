import numpy as np 
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix

#Specificidade
def specificity_loss_func(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return tn/(tn+fp)

def print_results(auc,acc,f1,specificity,sensitivity,precision):
      print('AUC: ' + str(auc))
      print('ACC: ' + str(acc))
      print('F1: ' + str(f1))
      print('Specificity: ' + str(specificity))
      print('Sensitivity: ' + str(sensitivity))
      print('Precision: ' + str(precision))

#SVM Padrãozão
clf = svm.SVC(gamma='scale')
#Lendo dados
data = pd.read_csv("radiomics.csv", usecols= lambda column : column not in ["id","malignancy","class"])
n_samples = len(data.index)
#Normalizando
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = min_max_scaler.fit_transform(data)
#Capturando as classe
classes = pd.read_csv("radiomics.csv", usecols=["class"])
classes = classes.values.ravel()
#Transformando em 0 e 1 para os scores
classes = [c.replace('BENIGN','0') for c in classes]
classes = [c.replace('MALIGNANT','1') for c in classes]
classes = [int(c) for c in classes]

clf.fit(data_scaled[:4*n_samples//5],classes[:4*n_samples//5])
## 10-fold
scores = {'AUC': 'roc_auc','ACC': 'accuracy','F1': 'f1','Sensitivity': 'recall','Precision': 'precision','Specificity': make_scorer(specificity_loss_func, greater_is_better=True)}
results = cross_validate(clf, data_scaled,classes, scoring=scores,cv=10,return_estimator=True)
#Calculando a média dos resultados
AUC = np.average(results['test_AUC'])
F1 = np.average(results['test_F1'])
ACC = np.average(results['test_ACC'])
Sensitivity = np.average(results['test_Sensitivity'])
Precision = np.average(results['test_Precision'])
Specificity = np.average(results['test_Specificity'])
#Printando as médias
print_results(AUC,ACC,F1,Specificity,Sensitivity,Precision,)


