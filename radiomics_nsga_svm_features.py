import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from platypus import NSGAII, Problem, Binary, nondominated
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
        print(solution.objectives)


def calculate_relevancy(results, objective_idx, threshold = 0, max_variables = 10):
    # results: Results of algorithm execution
    # objective_idx: 1 - Sensitivity, 2 - Specificity
    # threshold: Minimum condition to consider a model
    # max_variables: Amount of features in result plot

    limit = -1
    # Sorting results by parameter specified in objective_idx and selecting max index of solutions which archieves threshold
    results.sort(key=lambda s: s.objectives[objective_idx], reverse=True)
    for r in results:
        if r.objectives[objective_idx] >= threshold:
            limit += 1
        else:
            break

    num_of_variables = len(results[0].variables[0])
    variables_frequency = np.zeros(num_of_variables) # Array of counters for each feature in selected models

    # Counting occurrence of each feature in each model
    for i in range(limit + 1):
        r = results[i]
        for j in range(num_of_variables):
            if r.variables[0][j]:
                variables_frequency[j] += 1

    variables_frequency = variables_frequency * 100 / (limit + 1) # Transforming results in percentage

    data = pd.read_csv('radiomics.csv', usecols=lambda column: column not in ["class"])
    variable_names = data.columns.tolist()

    # Creating dict for ordering features names according with its percentages
    hash_variables_f = {}
    for i in range(num_of_variables):
        hash_variables_f[variable_names[i]] = variables_frequency[i]

    hash_variables_f = sorted(hash_variables_f.items(), key=lambda item: item[1], reverse=True)
    hash_variables_f = hash_variables_f[0:max_variables]

    var_list = [e[0] for e in hash_variables_f]
    y_pos = np.arange(len(var_list))
    relevancy = [e[1] for e in hash_variables_f]


    # Ploting results in a horizontal bar chart
    _, ax = plt.subplots(figsize=(11, 11))
    ax.barh(y_pos, relevancy, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(var_list)
    ax.invert_yaxis()
    ax.set_xlabel('Relevancy')
    if objective_idx == 0:
        ax.set_title('Best Variables - Sensitivity')
    else:
        ax.set_title('Best Variables - Specificity')
    plt.show()


if __name__ == "__main__":

    algorithm = NSGAII(SVM(), population_size=30)
    algorithm.run(100)

    nondominated_results = nondominated(algorithm.result)    

    # prints results
    fig1 = plt.figure(figsize=[11, 11])
    plt.scatter([s.objectives[0] for s in nondominated_results],
                [s.objectives[1] for s in nondominated_results])
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.xlabel("Sensitivity")
    plt.ylabel("Specificity")
    plt.show()

    calculate_relevancy(nondominated_results, 0, 0.81, 15)
    calculate_relevancy(nondominated_results, 1, 0.83, 15)
