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



def main(index_GP):

	print("Number of rows following these rules:", len(index_GP))

	y_val = y_test.loc[index_GP]
	SI_val = y_val['Stability'].value_counts()[1] / len(y_val) * 100
	print("The Stability Index on VALIDATION SET (sampled from TEST SET) is: SI =",round(SI_val, 4), "%")

	SI_test = len(y_test[y_test['Stability']==1])/len(y_test) * 100
	print("The Stability Index on TEST SET is: SI =",round(SI_test, 4), "%")



def catboost():

	index_GP = X_test[(X_test['Gamma_GLUDC'] <= -1.277) 
                  & (X_test['sigma_km_product1_ICDHxm'] > 0.056)
                  & (X_test['sigma_km_product2_GS'] > -0.802)
                  & (X_test['sigma_km_substrate2_ILETAm'] <= 1.365)
                  & (X_test['sigma_km_substrate1_ADK1'] <= 1.551)].index
	
	return index_GP              



def logreg():

	index_GP = X_test[(X_test['Gamma_GLUDC'] <= -1.277) 
                  & (X_test['sigma_km_product1_CHORS'] <= -1.075)].index

	return index_GP


def svc():

	index_GP = X_test[(X_test['Gamma_GLUDC'] <= -1.277) 
                  & (X_test['sigma_km_substrate1_2OXOADPTm'] <= 0.494)
                  & (X_test['sigma_km_substrate26_LMPD_s_0450_c_1_256'] <= 0.398)
                  & (X_test['sigma_km_substrate2_GK1'] <= 1.581)].index

	return index_GP              


def xgbclass():
	
	index_GP = X_test[(X_test['Gamma_GLUDC'] <= -1.277) 
                  & (X_test['sigma_km_product1_ICDHxm'] > 0.056)
                  & (X_test['sigma_km_product2_GS'] > -0.802)
                  & (X_test['sigma_km_substrate2_ILETAm'] <= 1.365)
                  & (X_test['sigma_km_substrate1_ADK1'] <= 1.551)].index

	return index_GP              


def dectree():

	index_GP = X_test[(X_test['Gamma_HCO3E'] > 1.843) 
                  & (X_test['sigma_km_substrate2_GAPD'] <= 0.08)].index

	return index_GP              


def frst():
	
	index_GP = X_test[(X_test['Gamma_GLUDC'] <= -1.277) 
                  & (X_test['sigma_km_product1_ICDHxm'] > 0.056)
                  & (X_test['sigma_km_product2_GS'] > -0.802)
                  & (X_test['sigma_km_substrate2_ILETAm'] <= 1.365)
                  & (X_test['sigma_km_substrate1_ADK1'] <= 1.551)].index

	return index_GP              


'''
Call main function to print results
'''
print('catboost')
index_GP = catboost()
main(index_GP)

print('logreg')
index_GP = logreg()
main(index_GP)

print('svc')
index_GP = svc()
main(index_GP)

print('xgbclass')
index_GP = xgbclass()
main(index_GP)

print('dectree')
index_GP = dectree()
main(index_GP)




def skoperules():

	index_GP = X_test[(X_test['Gamma_FBA'] <= -0.9118080735206604 ) 
                  & (X_test['sigma_km_product1_ALCD26xi'] > -0.26216644048690796 )
                  & (X_test['sigma_km_substrate1_ASPTA'] <= 1.1930022835731506)].index

	return index_GP              


print('Skope Rules')
index_GP = skoperules()
main(index_GP)




'''
catboost
Number of rows following these rules: 9
The Stability Index on VALIDATION SET (sampled from TEST SET) is: SI = 55.5556 %
The Stability Index on TEST SET is: SI = 19.8473 %


logreg
Number of rows following these rules: 14
The Stability Index on VALIDATION SET (sampled from TEST SET) is: SI = 71.4286 %
The Stability Index on TEST SET is: SI = 19.8473 %


svc
Number of rows following these rules: 18
The Stability Index on VALIDATION SET (sampled from TEST SET) is: SI = 61.1111 %
The Stability Index on TEST SET is: SI = 19.8473 %


xgbclass
Number of rows following these rules: 9
The Stability Index on VALIDATION SET (sampled from TEST SET) is: SI = 55.5556 %
The Stability Index on TEST SET is: SI = 19.8473 %


dectree
Number of rows following these rules: 11
The Stability Index on VALIDATION SET (sampled from TEST SET) is: SI = 63.6364 %
The Stability Index on TEST SET is: SI = 19.8473 %


Skope Rules
Number of rows following these rules: 11
The Stability Index on VALIDATION SET (sampled from TEST SET) is: SI = 90.9091 %
The Stability Index on TEST SET is: S'''
