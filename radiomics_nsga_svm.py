import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from platypus import NSGAII, Problem, Binary, Hypervolume, calculate, display
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


if __name__ == "__main__":
    experiment = 'radiomics_nsga_svm'
    X, Y = read_data('radiomics.csv')
    
    num_iter, generations_amount, pop = 5, 10, 10

    hypervolumes = [[] for i in range(num_iter)]
    hypervolumes_std = []
    hypervolumes_mean = []
    
    results = {'acc_mean': [],'acc_std': [], 'spec_mean': [], 'spec_std': [], 'sens_mean': [], 'sens_std': [], 'f1_score_mean': [], 'f1_score_std': [], 'auc_mean': [], 'auc_std': []}
    
    for i in range(num_iter):
        algorithm = NSGAII(SVM(), population_size=pop)
        
        for j in tqdm(range(generations_amount), desc="Iteration " + str(i+1)):
            algorithm.step()
            # Defining structure to pass as parameter to class Hypervolume
            nsga_results =  OrderedDict()
            nsga_results["NSGAII"] = {}
            nsga_results["NSGAII"]["SVM"] = [algorithm.result]

            # calculate the hypervolume indicator
            hyp = Hypervolume(minimum=[0, 0, 0], maximum=[1, 1, 1])
            hyp_result = calculate(nsga_results, hyp)
            hypervolume = np.mean(list(hyp_result["NSGAII"]["SVM"]["Hypervolume"]))
            hypervolumes[i].append(hypervolume)
        
        nondominated_results=nondominated(algorithm.result)
        # display(hipervolume, ndigits=3)
        solution = nondominated_results[0]
        features = solution.variables[0]
        for s in nondominated_results:
            if abs(s.objectives[0] - s.objectives[1]) < abs(solution.objectives[0] - solution.objectives[1]):
                solution = s
                features = s.variables[0]

        model = get_model(probability=True)
        result = validate(model, X[:, features], Y, plot = False)
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

    gen_std = np.std(hypervolumes,axis=0)
    gen_mean = np.mean(hypervolumes,axis=0)

    dict_results = {'mean':gen_mean, 'std':gen_std}
    
    outfile = open('results/' + experiment + '.pickle','wb')    
    pickle.dump(dict_results, outfile)
    outfile.close()
                
    df = pd.DataFrame(results)
    df.to_csv('results/' + experiment + '.csv')