import numpy as np
import pandas as pd

from sklearn import svm, preprocessing, metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

import matplotlib.pylab as plt
from scipy import interp

def specificity_loss_func(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return tn/(tn+fp)


def print_results(results):
    print("Accuracy: %.3f%% (+/- %.3f%%)" %
          (np.mean(results['acc']), np.std(results['acc'])))
    print("Specificity: %.3f%% (+/- %.3f%%)" %
          (np.mean(results['spec']), np.std(results['spec'])))
    print("Sensitivity: %.3f%% (+/- %.3f%%)" %
          (np.mean(results['sens']), np.std(results['sens'])))
    print("F1-score: %.3f%% (+/- %.3f%%)" %
          (np.mean(results['f1_score']), np.std(results['f1_score'])))
    print("AUC: %.3f (+/- %.3f)" %
          (np.mean(results['auc']), np.std(results['auc'])))


# SVM
clf = svm.SVC(gamma='scale', probability=True, class_weight='balanced')

# Reading data
data = pd.read_csv("radiomics.csv", usecols=lambda column: column not in [
                   "id", "malignancy", "class"])

classes = pd.read_csv("radiomics.csv", usecols=["class"])
classes = classes.values.ravel()

# Minmax scaling
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = min_max_scaler.fit_transform(data)

# Converting class into int
classes = [c.replace('BENIGN', '0') for c in classes]
classes = [c.replace('MALIGNANT', '1') for c in classes]
classes = [int(c) for c in classes]
classes = np.asarray(classes)

### Validation and evaluation

# Generating folds
cv = StratifiedKFold(n_splits=10, shuffle=True)

# Results dict
results = {'acc': [], 'spec': [], 'sens': [], 'f1_score': [], 'auc': []}

i = 0
#Para figura
fig1 = plt.figure(figsize=[11, 11])
tprs, aucs = [], []
mean_fpr = np.linspace(0, 1, 100)

# Cross-validation
for train, test in cv.split(data_scaled, classes):
    clf = clf.fit(data_scaled[train], classes[train])

    # Generating predictions
    predicted = clf.predict(data_scaled[test])
    predicted_proba = clf.predict_proba(data_scaled[test])

    results['acc'].append(accuracy_score(classes[test], predicted))
    results['f1_score'].append(f1_score(classes[test], predicted))
    results['spec'].append(specificity_loss_func(classes[test], predicted))
    results['sens'].append(recall_score(classes[test], predicted))
    results['auc'].append(roc_auc_score(classes[test], predicted_proba[:, 1]))

    # Composing ROC curve 
    i += 1
    
    fpr, tpr, t = metrics.roc_curve(classes[test], predicted_proba[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    #roc_auc = metrics.auc(fpr, tpr)
    roc_auc = results['auc'][i-1]
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

mean_tpr = np.mean(tprs, axis=0)
mean_auc = metrics.auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")

plt.show()

print_results(results)
