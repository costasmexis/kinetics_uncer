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
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier
from sklearn import tree
from sklearn.tree import _tree

import xgboost as xgb

SEED = 42


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



def main():

	print("Number of rows following these rules:", len(index_GP))

	y_val = y_test.loc[index_GP]
	SI_val = y_val['Stability'].value_counts()[1] / len(y_val)
	print("The Stability Index on VALIDATION SET (sampled from TEST SET) is: SI =",round(SI_val, 4), "%")

	SI_test = len(y_test[y_test['Stability']==1])/len(y_test) * 100
	print("The Stability Index on TEST SET is: SI =",round(SI_test, 4), "%")



'''
Rule:

	if (Gamma_GLUDC <= -1.277) and (sigma_km_product1_ICDHxm > 0.056) 
	and (sigma_km_product2_GS > -0.802) and (sigma_km_substrate2_ILETAm <= 1.365) 
	and (sigma_km_substrate1_ADK1 <= 1.551) then class: 1 (proba: 100.0%) | based on 25 samples

'''

index_GP = X_test[(X_test['Gamma_GLUDC'] <= -1.277) 
                  & (X_test['sigma_km_product1_ICDHxm'] > 0.056)
                  & (X_test['sigma_km_product2_GS'] > -0.802)
                  & (X_test['sigma_km_substrate2_ILETAm'] <= 1.365)
                  & (X_test['sigma_km_substrate1_ADK1'] <= 1.551)].index




main()


