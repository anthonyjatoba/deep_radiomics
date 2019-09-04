from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
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
            self.model, X, self.Y, scoring=scores, cv=3, return_estimator=True, n_jobs=3)
        solution.objectives[:] = np.mean(results['test_AUC'])        
        #print(solution.objectives)

if __name__ == "__main__":
    generations = 5
    num_iter = 5
    pop = 5
    ms = 5
    lw = 2
    capsize = 3
    elw = 0.5
    gen_scores = [[] for i in range(num_iter)]
    gen_std = []
    gen_mean = []
    results = {'acc_mean': [],'acc_std': [], 'spec_mean': [], 'spec_std': [], 'sens_mean': [], 'sens_std': [], 'f1_score_mean': [], 'f1_score_std': [], 'auc_mean': [], 'auc_std': []}
    x = np.arange(1, generations+1, 1)
    X, Y = read_data('radiomics.csv')
    for i in tqdm(range(num_iter)):
        #Reset alogirthm each iteration
        algorithm = GeneticAlgorithm(SVM(), population_size=pop)
        for j in tqdm(range(generations)):
            algorithm.step()
            gen_scores[i].append(algorithm.fittest.objectives[:])
    
        best_solution = algorithm.fittest
        features = best_solution.variables[0]
        model = get_model(probability=True)
        result = validate(model, X[:, features], Y, plot=False)
        
        results['acc_mean'].append(np.mean(result['acc']))
        results['acc_std'].append(np.std(result['acc']))
        results['spec_mean'].append(np.mean(result['spec']))
        results['spec_std'].append(np.std(result['spec']))
        results['sens_mean'].append(np.mean(result['sens']))
        results['sens_std'].append(np.std(result['sens']))
        results['f1_score_mean'].append(np.mean(result['f1_score']))
        results['f1_score_std'].append(np.std(result['f1_score']))
        results['auc_mean'].append(np.mean(result['auc']))
        results['auc_std'].append(np.std(result['auc']))


    gen_std = np.std(gen_scores,axis=0)
    gen_mean = np.mean(gen_scores,axis=0)
    # Evaluating best model
    df = pd.DataFrame(results)
    df.to_csv('results_ga.csv')

    
    
    plt.errorbar(x, np.array(gen_mean), np.array(gen_std),ms=ms, lw=lw, marker="o", capsize=capsize, ecolor="blue", elinewidth=elw, label="AUC")
    plt.show()
    # # Plotting fitness vs generations
    # plt.figure(figsize=[11, 11])
    # plt.title("Fitness vs Generations")

    # plt.xlabel("Generations")
    # plt.ylabel("Fitness (AUC)")
    # plt.plot(gen_scores)
    # plt.show()
    # Evaluating best model
    # features = best_solution.variables[0]

    # model = get_model(probability=True)

    # X, Y = read_data('radiomics.csv')
    # results = validate(model, X[:, features], Y)
    # print_summary(results)