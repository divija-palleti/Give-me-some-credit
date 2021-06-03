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
