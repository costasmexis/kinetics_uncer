import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier
from sklearn import tree
from sklearn.tree import _tree

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

import xgboost as xgb
import lightgbm as lgb

# Import skope-rules
import six
import sys

sys.modules['sklearn.externals.six'] = six
from skrules import SkopeRules


# ===================
# Input variables
# ===================
SEED = 42
filename = '../models/test_35%'

def print_scores(y_true, y_pred):
    print('ROCAUC score:',roc_auc_score(y_true, y_pred).round(4))
    print('Accuracy score:',accuracy_score(y_true, y_pred).round(4))
    print('F1 score:',f1_score(y_true, y_pred).round(4))
    print('Precision score:',precision_score(y_true, y_pred).round(4))
    print('Recall:',recall_score(y_true, y_pred).round(4))

def run_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return score, y_pred

def tune_model(model, param_grid, n_iter, X_train, y_train):
    grid = RandomizedSearchCV(model, param_grid, verbose=20,
        scoring='roc_auc', cv=3, n_iter=n_iter)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    return best_model



'''

LOAD FILES

'''
df = pd.read_csv('../data/Parameters_90%stability.csv')
df = df.drop(['Unnamed: 0'], axis = 1)

# Load X and Y
X = df.drop(['Stability'], axis = 1)
y = df['Stability']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.35,
                                                stratify=y, random_state=SEED)

y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

class_names = y_train['Stability'].unique().astype(str)
feature_names = x_train.columns.values

scaler = StandardScaler()

X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

X_train = pd.DataFrame(X_train, columns=x_train.columns)
X_train.index = x_train.index

X_test = pd.DataFrame(X_test, columns=x_test.columns)
X_test.index = x_test.index


# Train a skope-rules-boosting classifier
skope_rules_clf = SkopeRules(feature_names=feature_names, 
                             random_state=SEED,
                             # max_features='auto',
                             n_estimators=30,
                             recall_min=0.05, precision_min=0.9,
                             max_samples=0.7,
                             max_depth_duplication= 4, 
                             max_depth = 5, 
                             verbose=2)

skope_rules_clf.fit(X_train.values, y_train.values.reshape(-1,))


print(str(len(skope_rules_clf.rules_)) + ' rules have been built with ' +
      'SkopeRules.\n')

for i_rule, rule in enumerate(skope_rules_clf.rules_[:4]):
    print(rule[0])


def rules_to_txt(rules, filename):

    # open file in write mode
    with open(r'../rules/'+filename, 'w') as fp:
        for item in rules:
            # write each item on a new line
            fp.write("%s\n" % item)

rules_to_txt(skope_rules_clf.rules_, 'SkopeRules.txt')