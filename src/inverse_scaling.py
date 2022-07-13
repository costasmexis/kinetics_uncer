import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
                                                        stratify=y,
                                                        random_state=SEED)

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


'''

Create scaler_df

'''
scaler_df = pd.DataFrame(columns=scaler.feature_names_in_, index=['mean', 'std'])
scaler_df.loc['mean'] = scaler.mean_
scaler_df.loc['std'] = np.sqrt(scaler.var_)
scaler_df

'''

def inverse_scaling

'''
def inverse_scaling(col, elm):

    mu = scaler_df[col].loc['mean']
    sigma = scaler_df[col].loc['std']

    return elm * sigma + mu


el = inverse_scaling('Gamma_IPPS', 1.224024)
print(el)

