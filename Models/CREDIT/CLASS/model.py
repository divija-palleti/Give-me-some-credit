import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size1)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size1, hidden_size2)
    self.fc3 = nn.Linear(hidden_size2, hidden_size1)
    self.fc4 = nn.Linear(hidden_size1, num_classes)
    self.sig = nn.Sigmoid()
    self.soft = nn.Softmax(dim=2)
  
  def forward(self,x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    out = self.relu(out)
    out = self.fc3(out)
    out = self.sig(out)
    out = self.relu(out)
    out = self.fc4(out)
    # out = self.sig(out)
    return out

  def prob_predict(self, data):
    output = self.forward(data)
    print(output, "before sigmoid")
    output = self.soft(output)
    return output