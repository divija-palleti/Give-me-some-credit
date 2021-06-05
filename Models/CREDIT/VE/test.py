import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing.data import MinMaxScaler

from model import *

def main():

    batch_size = 64

    y_train = np.genfromtxt('../labels.csv',delimiter=',', dtype=float)
    x_train = np.genfromtxt('../free.csv', delimiter=',', dtype=float)
    print(x_train.shape)
    x_train = x_train.astype(float)
    scaler = StandardScaler()
    x_train =  scaler.fit_transform(x_train)
    

    # Parameters of the VAE
    d = 4 # latent space 
    D = input_dim = x_train.shape[1]
    activFunName = 'relu'  
    activations_list = {
        'softplus': nn.Softplus(),
        'tanh': nn.Tanh(),
        'relu': nn.ReLU()
    }
    activFun = activations_list[activFunName]
    H1 = 64
    H2 = 128

    model = VAE_model(d, D, H1, H2, activFun)
    model_path = './results/model_VE_minmax_1.pt'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    np.savetxt('./results/train_input_scaled.txt',scaler.inverse_transform(x_train) , delimiter=',', fmt='%f')
    np.savetxt('./results/train_input.txt',(x_train) , delimiter=',', fmt='%f')

    
    k  = torch.from_numpy(x_train)
    k = k.type(torch.FloatTensor)

    # MU_X_eval, Z_ENC_eval, MU_Z_eval, LOG_VAR_Z_eval = model(k)

    MU_X_eval, Z_ENC_eval, MU_Z_eval = model(k)
    MU_X_eval = MU_X_eval.detach().numpy()
    np.savetxt('./results/train_output_scaled.txt', scaler.inverse_transform(MU_X_eval), delimiter=',', fmt='%f')
    np.savetxt('./results/train_output.txt', MU_X_eval, delimiter=',', fmt='%f')


    print("llll")
    # MU_X_eval, Z_ENC_eval, MU_Z_eval, LOG_VAR_Z_eval = model(torch.from_numpy(x_test))
    # MU_X_eval, Z_ENC_eval, MU_Z_eval = model(torch.from_numpy(x_test))
    # MU_X_eval = MU_X_eval.detach().numpy()
    # np.savetxt('./results/test.txt',MU_X_eval , delimiter=',', fmt='%f')


if __name__ == "__main__":
    main()