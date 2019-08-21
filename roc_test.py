import numpy as np 
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import matplotlib.pylab as plt
from scipy import interp
#Specificidade
def specificity_loss_func(y,y_pred):
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
clf = svm.SVC(gamma='scale',probability=True)
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
classes = [c.replace('BENIGN','1') for c in classes]
classes = [c.replace('MALIGNANT','0') for c in classes]
classes = [int(c) for c in classes]
classes = np.asarray(classes)

#Divisor
cv = StratifiedKFold(n_splits=10,shuffle=True)
#Arrays de metricas
acc = np.zeros(10)
f1 = np.zeros(10)
auc = np.zeros(10)
sensitivity = np.zeros(10)
specificity = np.zeros(10)
precision = np.zeros(10)
i = 0
#Para figura
fig1 = plt.figure(figsize=[11,11])
tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
#Scores
scores = {'AUC': metrics.roc_auc_score,'ACC': metrics.accuracy_score,'F1': metrics.f1_score,'Sensitivity': metrics.recall_score,'Precision': metrics.precision_score,'Specificity': specificity_loss_func}
for train,test in cv.split(data_scaled,classes):
    clf = clf.fit(data_scaled[train],classes[train])
    predicted = clf.predict(data_scaled[test])
    #Calculando os scores
    predicted_proba = clf.predict_proba(data_scaled[test])
    acc[i] = scores['ACC'](classes[test],predicted)
    f1[i] = scores['F1'](classes[test],predicted)
    sensitivity[i] = scores['Sensitivity'](classes[test],predicted)
    precision[i] = scores['Precision'](classes[test],predicted)
    specificity[i] = scores['Specificity'](classes[test],predicted)
    auc[i] = scores['AUC'](classes[test],predicted_proba[:, 1])
    i+=1
    #Coisas do gráfico
    fpr, tpr, t = metrics.roc_curve(classes[test], predicted_proba[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    #Qual dos dois usar ?
    roc_auc = metrics.auc(fpr, tpr)
    #roc_auc = auc[i-1]
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))


mean_tpr = np.mean(tprs, axis=0)
mean_auc = metrics.auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")

print_results(np.average(auc),np.average(acc),np.average(f1),np.average(specificity),np.average(sensitivity),np.average(precision))
plt.show()

