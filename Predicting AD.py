#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:37:29 2019

@author: samuelghatan
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
import operator
import matplotlib.pyplot as plt

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

# Hyper parameters were tuned using SciKit-Learns Randomised Search CV 
rf = RandomForestClassifier(n_estimators = 400, min_samples_split = 10, min_samples_leaf = 2, max_features = 'auto', max_depth = 40, bootstrap = False)
rfc.fit(train_sample[soma_columns],train_sample['diag_AD'])
score = rfc.score(test_sample[soma_columns], test_sample['diag_AD'])
print(round(score,2))

# Standard model produced accuracy around 50-55% orginally
# By removing MCI from the sample while also creating an even training sample of 100/100 AD/CTL the accuracy was increased to around 68% consistantly.

""" Feature Importance

Next I would like to optimize the model further by selecting the most predictive proteins are predictiors, 
In addition to this for prospective clinical application, it is desirable to keep the number of proteins as low as possible while maintaining a high performance.
Thus I will try and limit the number of predictive proteins to around 10 """

# The snippet below will retrieve the feature importances from the model and make them into a DataFrame.
feature_importances = pd.DataFrame(rfc.feature_importances_, index = train_sample[soma_columns].columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

top_20 = feature_importances.index.values[:20]
top_protein = 0
for i in range(1,100):
    train_sample = pd.concat([AD.sample(100),CTL.sample(100)],axis=0)
    test_sample = protein.loc[~protein.index.isin(train_sample.index)]
    test_sample = test_sample[test_sample.diag_MCI == 0]
    cols = top_20[:13]
    rfc.fit(train_sample[cols],train_sample['diag_AD'])
    score = rfc.score(test_sample[cols], test_sample['diag_AD'])
    top_protein += score
print(top_protein/99)

# The optimimum number is 13 with an average score of 69%
cols = top_20[:13]
rfc.fit(train_sample[cols],train_sample['diag_AD'])
score = rfc.score(test_sample[cols], test_sample['diag_AD'])
top_feature_importances = feature_importances = pd.DataFrame(rfc.feature_importances_, index = train_sample[cols].columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(top_feature_importances)