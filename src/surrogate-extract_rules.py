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


# ===================
# Input variables
# ===================
SEED = 42
filename = '../models/test_35%'


def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


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




def black_box(model, model_name, param_grid):

    try:

        # load the model from disk
        loaded_model = pickle.load(open(filename+'/'+ model_name, 'rb'))
        y_pred = loaded_model.predict(X_test)

        return loaded_model, y_pred

    except FileNotFoundError:

        if(param_grid == None):

            score, y_pred = run_model(model, X_train, y_train.values.ravel(),X_test, y_test.values.ravel())
            pickle.dump(model, open(filename+'/'+ model_name, 'wb'))
            return model, y_pred

        else:

            best_model = tune_model(model, param_grid, 300, X_train, y_train.values.ravel())
            score, y_pred = run_model(best_model, X_train, y_train.values.ravel(),
                X_test, y_test.values.ravel())

            pickle.dump(best_model, open(filename+'/'+ model_name, 'wb'))

            return best_model, y_pred


def surrogate(blackbox_model):

    y_pred_train = blackbox_model.predict(X_train)

    surrogate = DecisionTreeClassifier(random_state=SEED)

    surrogate.fit(X_train, y_pred_train)

    return surrogate


def extract_rules(surrogate):

    rules = get_rules(surrogate, feature_names=feature_names,
                  class_names=class_names)

    return rules


def rules_to_txt(rules, filename):

    # open file in write mode
    with open(r'../rules/'+filename, 'w') as fp:
        for item in rules:
            # write each item on a new line
            fp.write("%s\n" % item)

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


# =====================================
# Scaler
# =====================================
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)
pickle.dump(scaler, open('../scaler/scaler.pkl', 'wb'))


X_train = pd.DataFrame(X_train, columns=x_train.columns)
X_train.index = x_train.index

X_test = pd.DataFrame(X_test, columns=x_test.columns)
X_test.index = x_test.index


'''

SMOTE / Undersampling

'''
def smote(X, y):
    sm = SMOTE(random_state=SEED)
    X_res, y_res = sm.fit_resample(X, y)

    return X_res, y_res


def under(X, y):
    print("...Undersampling...")
    undersample = EditedNearestNeighbours(n_neighbors=3)
    X_res, y_res = undersample.fit_resample(X, y)

    return X_res, y_res


'''

Models

'''
def catboost(model_name='catboost_model.sav'):
    catboost, y_catboost = black_box(CatBoostClassifier(random_state=SEED), model_name, None)
    surrogate_catboost = surrogate(catboost)
    rules_catboost = extract_rules(surrogate_catboost)
    rules_to_txt(rules_catboost, 'rules_catboost.txt')


def logreg(model_name='logreg_model.sav'):
    log_reg, y_logreg = black_box(LogisticRegression(max_iter=100000), model_name, None)
    surrogate_logreg = surrogate(log_reg)
    rules_logreg = extract_rules(surrogate_logreg)
    rules_to_txt(rules_logreg, 'rules_logreg.txt')


def svc(model_name='svr_model.sav'):
    param_grid_svc = {'C': [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 1, 1.5, 2, 2.5, 3, 5, 10, 12, 20, 25, 50],
                'gamma': [0.002, 0.003, 0.004, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
                'kernel': ['rbf', 'linear']
    }

    svr, y_svr = black_box(SVC(random_state=SEED), model_name, param_grid_svc)
    surrogate_svr = surrogate(svr)
    rules_svr = extract_rules(surrogate_svr)
    rules_to_txt(rules_svr, 'rules_svr.txt')


def dectree(model_name='decisiontree_model.sav'):
    param_grid_dt = {
        'max_depth': [2, 3, 5, 10, 20],
        'min_samples_leaf': [5, 10, 20, 50, 100],
        'criterion': ["gini", "entropy"],
        'max_features': ['auto', 'sqrt', None]

    }

    dt, y_dt = black_box(DecisionTreeClassifier(random_state=SEED), model_name, param_grid_dt)
    surrogate_dt = surrogate(dt)
    rules_dt = extract_rules(surrogate_dt)
    rules_to_txt(rules_dt, 'rules_dt.txt')


def forest(model_name='forest_model.sav'):
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    param_grid_forest = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}    


    frst, y_frst = black_box(RandomForestClassifier(bootstrap=True, max_depth=50, min_samples_leaf=1, min_samples_split=2, n_estimators=1400, random_state=SEED), model_name, None)
    surrogate_frst = surrogate(frst)
    rules_frst = extract_rules(surrogate_frst)
    rules_to_txt(rules_frst, 'rules_frst.txt')


def xgbclass(model_name='xgb_model.sav'):
    param_grid_xgb = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
                     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
                     "min_child_weight" : [ 1, 3, 5, 7 ],
                     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
                     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }

    bst, y_bst = black_box(xgb.XGBClassifier(random_state=SEED), model_name, param_grid_xgb)
    surrogate_bst = surrogate(bst)
    rules_bst = extract_rules(surrogate_bst)
    rules_to_txt(rules_bst, 'rules_bst.txt')







# ==================
# Main
# ==================
# xgbclass()
# catboost()
# logreg()
# svc()
# dectree()
# forest()


