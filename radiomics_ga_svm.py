import numpy as np

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from platypus import GeneticAlgorithm, Problem, Binary
from radiomics_all_svm import specificity_loss_func, print_summary, read_data, validate, get_model


class SVM(Problem):
    def __init__(self):
        super(SVM, self).__init__(1, 1)
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
            self.model, X, self.Y, scoring=scores, cv=3, return_estimator=True, n_jobs=-1)
        print(np.mean(results['test_AUC']))
        solution.objectives[:] = np.mean(results['test_AUC'])


if __name__ == "__main__":
    algorithm = GeneticAlgorithm(SVM(), population_size=10)
    algorithm.run(100)

    # Selecting the best solution
    best_solution = algorithm.result[0]
    for s in algorithm.result:
        if s.objectives[0] > best_solution.objectives[0]:
            best_solution = s

    features = best_solution.variables[0]

    model = get_model()

    X, Y = read_data('radiomics.csv')
    results = validate(model, X[:, features], Y)
    print_summary(results)
