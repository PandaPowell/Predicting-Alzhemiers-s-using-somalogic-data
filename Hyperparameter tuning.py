#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:42:05 2019

@author: samuelghatan
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import operator
from sklearn.model_selection import RandomizedSearchCV

pd.options.display.max_columns = 999

protein = pd.read_csv('Somalogic data.csv')
num_columns = list(protein.columns)
num_columns.remove('sample_ID')
num_columns.remove('Lab Code')

protein = protein[num_columns]

# Convert gender into catergorical variable

def convert_gender(gender):
    if gender == 'F':
        return 1
    elif gender == 'M':
        return 0

protein['gender'] = protein['gender'].apply(convert_gender)

# Convert multi-class diagnosis feature into binary using dummy variables

dummy_df = pd.get_dummies(protein["final_diag"], prefix="diag")
protein = pd.concat([protein, dummy_df], axis=1)

correlations = protein.corr()
correlations['diag_AD'].sort_values()

# Remove Nans
nans = protein.isnull().sum()
nans[nans > 0]

# For nans in baseline diag make same as final diag
index_num = protein[protein.baseline_diag.isnull()].index
protein['baseline_diag'].iloc[373] = 2

# for missing gender and apoe replace with most common value (mode)
protein['gender'] = protein['gender'].fillna(1).copy()
protein['APOE'] = protein['APOE'].fillna(0.0).copy()

soma_columns = num_columns
soma_columns.remove('age')
soma_columns.remove('APOE')
soma_columns.remove('baseline_diag')
soma_columns.remove('gender')
soma_columns.remove('final_diag')

# Split data into train and test
train = protein.sample(frac=0.75)
test = protein.loc[~protein.index.isin(train.index)]

# Split data by diagnosis
AD = protein[protein.diag_AD == 1]
CTL = protein[protein.diag_CTL == 1]
MCI = protein[protein.diag_MCI == 1]

# Creating training and test set containing only AD & MCI
train_sample = pd.concat([AD.sample(100),CTL.sample(100)],axis=0)
test_sample = protein.loc[~protein.index.isin(train_sample.index)]
test_sample = test_sample[test_sample.diag_MCI == 0]

# Random Hyper parameter grid
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_sample[soma_columns],train_sample['diag_AD'])
#score = rfc.score(test_sample[soma_columns], test_sample['diag_AD'])
#print(round(score,2))