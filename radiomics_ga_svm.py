import numpy as np
import pandas as pd

from sklearn import svm, preprocessing
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

import matplotlib.pylab as plt

from platypus import GeneticAlgorithm, Problem, Binary

from radiomics_all_svm import run_svm

def specificity_loss_func(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return tn/(tn+fp)


def print_results(results):
    print("Accuracy: %.3f (+/- %.3f)" %
          (np.mean(results['acc']), np.std(results['acc'])))
    print("Specificity: %.3f (+/- %.3f)" %
          (np.mean(results['spec']), np.std(results['spec'])))
    print("Sensitivity: %.3f (+/- %.3f)" %
          (np.mean(results['sens']), np.std(results['sens'])))
    print("F1-score: %.3f (+/- %.3f)" %
          (np.mean(results['f1_score']), np.std(results['f1_score'])))
    print("AUC: %.3f (+/- %.3f)" %
          (np.mean(results['auc']), np.std(results['auc'])))


class SVM(Problem):
    def __init__(self):
        super(SVM, self).__init__(1, 1)
        self.types[:] = Binary(122)
        
        # SVM
        self.clf = svm.SVC(gamma='scale', probability=False, class_weight='balanced')
        
        # Reading data
        data = pd.read_csv("radiomics.csv", usecols=lambda column: column not in [
           "id", "malignancy", "class"])
        
        classes = pd.read_csv("radiomics.csv", usecols=["class"])
        classes = classes.values.ravel()
        
        # Minmax scaling
        min_max_scaler = preprocessing.MinMaxScaler()
        self.data = min_max_scaler.fit_transform(data)
        
        # Converting class into int
        classes = [c.replace('BENIGN', '0') for c in classes]
        classes = [c.replace('MALIGNANT', '1') for c in classes]
        classes = [int(c) for c in classes]
        self.classes = np.asarray(classes)
        self.directions[:] = Problem.MAXIMIZE
    
    def evaluate(self, solution):
        x = solution.variables[:]
        # Selecting the columns
        data = self.data[:, x[0]]
                
        scores = {'AUC': 'roc_auc','ACC': 'accuracy','F1': 'f1','Sensitivity': 'recall','Precision': 'precision','Specificity': make_scorer(specificity_loss_func, greater_is_better=True)}
        results = cross_validate(self.clf, data, self.classes, scoring=scores, cv=10, return_estimator=True, n_jobs=4)

        solution.objectives[:] = np.mean(results['test_AUC'])
        print(solution.objectives)
      
algorithm = GeneticAlgorithm(SVM(), population_size=10)
algorithm.run(10)

best_solution = algorithm.result[0]

for s in algorithm.result:
  if s.objectives[0] > best_solution.objectives[0]:
    best_solution = s

run_svm("--- Features selected by GA ---", best_solution.variables[0])