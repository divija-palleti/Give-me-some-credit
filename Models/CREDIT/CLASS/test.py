import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import model as model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
batch_size = 256


hidden_size1 = 32
hidden_size2 = 64 # number of nodes at hidden layer
num_classes = 2 
num_epochs = 50 # number of times which the entire dataset is passed throughout the model
lr = 1e-3 *3 # size of step


y_train = np.genfromtxt('../labels.csv',delimiter=',', dtype=float)
y_train_class1 = np.genfromtxt('../labels_1.csv',delimiter=',', dtype=float)
x_train_1 = np.genfromtxt('../free.csv', delimiter=',', dtype=float)
x_train_2 = np.genfromtxt('../conditionals.csv', delimiter=',', dtype=float)
x_train = np.concatenate((x_train_1, x_train_2), axis=1)
print(x_train.shape)
x_train = x_train.astype(float)
scaler = MinMaxScaler()
x_train =  scaler.fit_transform(x_train)

input_size = data_dim = x_train.shape[1]

net = model.Net(input_size, hidden_size1, hidden_size2, num_classes)
model_path = './results/classification_model_batch_load.pt'
net.load_state_dict(torch.load(model_path))

x = torch.FloatTensor(x_train)
output = net.prob_predict(x)
_, predicted = torch.max(output,1)

y_true = y_train
y_pred = predicted.numpy()
precision_score = precision_score(y_true = y_true, y_pred= y_pred)
recall_score = recall_score(y_true = y_true, y_pred=y_pred)

print('precision_score of the model', precision_score)
print('recall_score of the model', recall_score)
