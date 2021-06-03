import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.data import MinMaxScaler

# Credit Data Processing
# For further info on constraints imposed below: see also appendix in Ustun et al (2018)
raw_df = pd.read_csv('./cs-training.csv')
processed_df = raw_df

# drop NAs & unnamed column & convert boolean to numeric
processed_df = processed_df.dropna()
processed_df = processed_df.drop(columns='Unnamed: 0')
processed_df = processed_df + 0 
processed_df = processed_df.loc[processed_df['age']<88]

# look at column names
print(processed_df.columns)
scalable = [ 'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']

scalable = [ 'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']

# scaler = MinMaxScaler()
# processed_df[scalable] =  scaler.fit_transform(processed_df[scalable])


# Labels, protected & free featuers
# labels
epsilon = 1e-4
# we clip 0 to avoid evaluation errors when using log normal likelihood

labels = processed_df[processed_df.columns[0]]
labels.columns = [processed_df.columns[0]]
# conditioning set/protected set
conditionals = processed_df[[processed_df.columns[2], processed_df.columns[10]]]
conditionals.columns = [processed_df.columns[2], processed_df.columns[10]]
# free features
free = processed_df.drop(columns=[processed_df.columns[0], processed_df.columns[2], processed_df.columns[10]])
free[free.columns[0]] = np.clip(free.values[:,0], epsilon, 1e20)
free[free.columns[2]] = np.clip(free.values[:,2], epsilon, 1e20)
free[free.columns[3]] = np.clip(free.values[:,3], epsilon, 1e20)
print(free.columns)

df = pd.DataFrame(free)

df.to_csv('free.csv', header=False, index=False)

df = pd.DataFrame(conditionals)

df.to_csv('conditionals.csv',  header=False, index=False)

df = pd.DataFrame(labels)

df.to_csv('labels.csv', header=False,  index=False)


# mport pandas as pd
# from scipy import stats
# import numpy as np

# filePath = 'Datasets/Give_Me_Some_Credit/'
# testFile = 'cs-test.csv'
# trainFile = 'cs-training.csv'

# # Read Data from csv
# test_df = pd.read_csv(filePath + testFile)
# train_df = pd.read_csv(filePath + trainFile)

# # Column 'SeriousDlqin2yrs' only used in training set as label
# try:
#     test_df.drop('SeriousDlqin2yrs', axis=1, inplace=True)
# except KeyError:
#     print('Column "SeriousDlqin2yrs" already droped')

# # drop rows with missing values
# test_df.dropna(axis=0, inplace=True)
# train_df.dropna(axis=0, inplace=True)

# # get rid of outliers
# continuous_cols = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
# 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines'
# ,'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']


# # # change labeling to be consistent with our notation
# # label_map = {
# #     0: 1,
# #     1: 0}
# # train_df['SeriousDlqin2yrs'] = train_df['
# # Martin Pawelczyk to Everyone (1:16 PM)
# # import torch
# # import numpy as np


# # from torch import nn

# # class ANN(nn.Module):
# #     def __init__(self, input_layer, hidden_layer_1, hidden_layer_2, output_layer, num_of_classes):
# #         """
# #         Defines the structure of the neural network
# #         :param input_layer: int > 0, number of neurons for this layer
# #         :param hidden_layer_1: int > 0, number of neurons for this layer
# #         :param hidden_layer_2: int > 0, number of neurons for this layer
# #         :param output_layer: int > 0, number of neurons for this layer
# #         :param num_of_classes: int > 0, number of classes
# #         """
# #         super().__init__()

# #         # number of input neurons
# #         self.input_neurons = input_layer

# #         # Layer
# #         self.input = nn.Linear(input_layer, hidden_layer_1)
# #         self.hidden_1 = nn.Linear(hidden_layer_1, hidden_layer_2)
# #         self.hidden_2 = nn.Linear(hidden_layer_2, output_layer)
# #         self.output = nn.Linear(output_layer, num_of_classes)

# #         # Activation
# #         self.relu = nn.ReLU()