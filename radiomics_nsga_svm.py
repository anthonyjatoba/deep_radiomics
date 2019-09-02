from tqdm import tqdm
import numpy as np
import matplotlib.pylab as plt

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from platypus import NSGAII, Problem, Binary, Hypervolume, calculate, display
from radiomics_all_svm import specificity_loss_func, read_data, get_model


class SVM(Problem):
    def __init__(self):
        super(SVM, self).__init__(1, 2)
        self.X, self.Y = read_data('radiomics.csv')
        self.types[:] = Binary(self.X.shape[1])
        self.model = get_model()
        self.directions[:] = Problem.MAXIMIZE

    def evaluate(self, solution):
        columns = solution.variables[:]

        # Selecting the columns
        X = self.X[:, columns[0]]

        scores = {'AUC': 'roc_auc', 'ACC': 'accuracy', 'F1': 'f1', 'Sensitivity': 'recall',
                  'Precision': 'precision', 'Specificity': make_scorer(specificity_loss_func, greater_is_better=True)}
        results = cross_validate(
            self.model, X, self.Y, scoring=scores, cv=3, return_estimator=True, n_jobs=3)

        solution.objectives[:] = [
            np.mean(results['test_Sensitivity']), np.mean(results['test_Specificity'])]
        #print(solution.objectives)


if __name__ == "__main__":

    from collections import OrderedDict
    algorithm = NSGAII(SVM(), population_size=10)
    generations_amount = 100

    hypervolumes = [0]
    for i in tqdm(range(generations_amount)):
        algorithm.step()
        # Defining structure to pass as parameter to class Hypervolume
        results =  OrderedDict()
        results["NSGAII"] = {}
        results["NSGAII"]["SVM"] = [algorithm.result]

        # calculate the hypervolume indicator
        hyp = Hypervolume(minimum=[0, 0, 0], maximum=[1, 1, 1])
        hyp_result = calculate(results, hyp)
        hypervolume = np.mean(list(hyp_result["NSGAII"]["SVM"]["Hypervolume"]))
        hypervolumes.append(hypervolume)
        # display(hipervolume, ndigits=3)

    fig1 = plt.figure(figsize=[11, 11])
    plt.plot([i for i in range(generations_amount+1)], hypervolumes)
    plt.xlabel("Generations")
    plt.ylabel("Hipervolume")
    plt.show()