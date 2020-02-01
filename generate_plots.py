# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_two_errorbar(dict1, dict2, title, xlabel, ylabel, label1, label2, output):
    ms = 5
    lw = 2
    capsize = 3
    elw = 0.5    
    x = np.arange(1, len(dict1['mean'])+1, 1)
    plt.figure(figsize=(12, 8))
    plt.errorbar(x, np.array(dict1['mean']), np.array(dict1['std']), ms=ms, lw=lw, marker="o", capsize=capsize, ecolor="blue", elinewidth=elw, label=label1)
    plt.errorbar(x, np.array(dict2['mean']), np.array(dict2['std']), ms=ms, ls="--", lw=lw, marker="s", capsize=capsize, ecolor="red", color="r", elinewidth=elw, label=label2, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right',prop={'size': 16})
    plt.tight_layout()
    plt.savefig('results/' + output + '.pdf', format="pdf")

    
ga_radiomics_path = 'results/radiomics_ga_svm.pickle'
ga_deep_radiomics_path = 'results/deep_radiomics_ga_svm.pickle'

nsga_radiomics_path = 'results/radiomics_nsga_svm.pickle'
nsga_deep_radiomics_path = 'results/deep_radiomics_nsga_svm.pickle'

infile = open(ga_radiomics_path,'rb')
ga_radiomics = pickle.load(infile)
infile.close()

infile = open(ga_deep_radiomics_path,'rb')
ga_deep_radiomics = pickle.load(infile)
infile.close()

infile = open(nsga_radiomics_path,'rb')
nsga_radiomics = pickle.load(infile)
infile.close()

infile = open(nsga_deep_radiomics_path,'rb')
nsga_deep_radiomics = pickle.load(infile)
infile.close()

plot_two_errorbar(ga_radiomics, ga_deep_radiomics, 'GA - AUC vs Generation', 'Generation', 'AUC', 'Radiomics', 'Deep + Radiomics', 'GA - AUC')
plot_two_errorbar(nsga_radiomics, nsga_deep_radiomics, 'NSGA-II - Hypervolume vs Generation', 'Generation', 'Hypervolume', 'Radiomics', 'Deep + Radiomics', 'NSGA-II - Hypervolume')