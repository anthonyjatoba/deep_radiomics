import numpy as np
import matplotlib.pylab as plt

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from platypus import NSGAII, Problem, Binary
from platypus.core import nondominated
from radiomics_all_svm import specificity_loss_func, print_summary, read_data, validate, get_model


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
        print(solution.objectives)


if __name__ == "__main__":
    algorithm = NSGAII(SVM(), population_size=10,  archive = [])
    algorithm.run(1000)

    # filter results
    nondominated_results = nondominated(algorithm.result)    
    # prints results
    fig1 = plt.figure(figsize=[11, 11])
    plt.scatter([s.objectives[0] for s in nondominated_results],
                [s.objectives[1] for s in nondominated_results])
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.xlabel("Sensitivity")
    plt.ylabel("Specificity")
    plt.title("Non dominated results")
    plt.show()

    # Selecting the solution with smallest difference between objectives
    solution = nondominated_results[0]
    features = solution.variables[0]

    for s in nondominated_results:
        if abs(s.objectives[0] - s.objectives[1]) < abs(solution.objectives[0] - solution.objectives[1]):
            solution = s
            features = s.variables[0]

    model = get_model(probability=True)

    X, Y = read_data('radiomics.csv')
    results = validate(model, X[:, features], Y)
    print_summary(results)
