import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import model as model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.metrics import precision_score
import pandas as pd
batch_size_1 = 200
batch_size_2 = 56 # all of these belong to class 1 ( minority)

#for now just trying with the training.csv

hidden_size1 = 32
hidden_size2 = 64 # number of nodes at hidden layer
num_classes = 2 
num_epochs = 50 # number of times which the entire dataset is passed throughout the model
lr = 1e-3 * 3 # size of step


y_train = np.genfromtxt('../labels.csv',delimiter=',', dtype=float)
y_train_class1 = np.genfromtxt('../labels_1.csv',delimiter=',', dtype=float)
x_train_1 = np.genfromtxt('../free.csv', delimiter=',', dtype=float)
x_train_2 = np.genfromtxt('../conditionals.csv', delimiter=',', dtype=float)
x_train_1_class_1 = np.genfromtxt('../free_1.csv', delimiter=',', dtype=float)
x_train_2_class_1 = np.genfromtxt('../conditionals_1.csv', delimiter=',', dtype=float)
x_train = np.concatenate((x_train_1, x_train_2), axis=1)

x_train = x_train.astype(float)
scaler = MinMaxScaler()
x_train =  scaler.fit_transform(x_train)

input_size = data_dim = x_train.shape[1]

net = model.Net(input_size, hidden_size1, hidden_size2, num_classes)
print(net)


optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# torch.optim.ASGD

train_loader = torch.utils.data.DataLoader(list(zip(x_train, y_train)), shuffle=True, batch_size=batch_size_1)
# test_loader = torch.utils.data.DataLoader(list(zip(x_train, y_train)), shuffle=True, batch_size=batch_size_1)

n_1 = y_train_class1.shape[0]
n_0 = y_train.shape[0] - n_1
        # w_0 = 1/ ((n_0 + n_1) / ( n_0))
        # w_1 = 1/ ((n_0 + n_1) / (n_1))
w_0 = (n_1)/(n_0 + 1e-5)
w_1 = (n_0)/(n_1 + 1e-5)
class_weights=torch.FloatTensor([1, 5]) #assigning weights

loss_function = nn.BCEWithLogitsLoss(pos_weight=class_weights)
# print(class_weights)
# print(kk)

loss_all = []

# class1_free = pd.read_csv('../free.csv')
# class1_free = pd.read_csv('../free.csv')

for epoch in range(num_epochs):

    for i ,(x, y) in enumerate(train_loader):

        index = np.random.choice(x_train_1_class_1.shape[0], batch_size_2, replace=False)
        # print(index)
        x_1_class_1 = x_train_1_class_1[index]
        x_2_class_1 = x_train_2_class_1[index]
        y_class_1 = y_train_class1[index]
        x_class_1 = np.concatenate((x_1_class_1, x_2_class_1), axis=1)

        x_class_1 = torch.FloatTensor(x_class_1)
        y_class_1 = torch.FloatTensor(y_class_1)

        x = torch.cat((x, x_class_1), axis = 0) #adding 56 samples of class 1
        y = torch.cat((y, y_class_1), axis = 0)  #adding 56 samples of class 1

        index_1 = np.random.choice(x.shape[0], 256, replace=True)
        x = x[index_1]
        y = y[index_1]


        # n_1 = torch.count_nonzero(y) + 1e-5
        # n_0 = y.shape[0] - n_1 + 1e-5
        # # w_0 = 1/ ((n_0 + n_1) / ( n_0))
        # # w_1 = 1/ ((n_0 + n_1) / (n_1))
        # w_0 = (n_1)/n_0
        # w_1 = (n_0)/n_1
        # class_weights=torch.FloatTensor([w_0, w_1])
        x=x.type(torch.FloatTensor)
        # print(class_weights)
        # print(kk)
        # loss_function = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        

        optimizer.zero_grad()
        outputs = net(x)

        # l = []
        target = torch.zeros((y.shape[0], 2))
        for z in range(y.shape[0]):
            if y[z]==0:
                target[z][0] = 1
            else:
                target[z][1] = 1
        
        # loss_function = nn.BCELoss(weight=class_weights) 
        loss = loss_function(outputs, target)
        loss_all.append(loss)
        # l.append(loss)
        loss.backward()
        optimizer.step()
    # loss_all.append(sum(l)/len(l))
    # print(loss_all)
    print('Epoch [%d/%d]' %(epoch+1, num_epochs))
plt.figure()
plt.plot(loss_all)
plt.legend(['all'], loc='upper left')
plt.savefig(f'./results/train')
plt.show()

#Evaluating the accuracy of the model

correct = 0
total = 0
for i ,(x, y) in enumerate(train_loader):
    # print(x.shape)
    x=x.type(torch.FloatTensor)
    s = x.shape[0]
    
    output = net(x)
    _, predicted = torch.max(output,1)
    # print(predicted,y)
    correct += (predicted == y).sum()
    # print(correct, "c")
    total += s

    print(total, "t")

x = torch.FloatTensor(x_train)
output = net(x)
_, predicted = torch.max(output,1)


acc = correct/total

print('Accuracy of the model', acc)

torch.save(net.state_dict(), './results/classification_model_batch_load.pt') 