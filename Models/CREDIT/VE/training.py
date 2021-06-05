from sklearn.preprocessing.data import MinMaxScaler
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from model import *

def main():

    batch_size = 256

    y_train = np.genfromtxt('../labels.csv',delimiter=',', dtype=float)
    x_train = np.genfromtxt('../free.csv', delimiter=',', dtype=float)
    print(x_train.shape)
    x_train = x_train.astype(float)
    scaler = MinMaxScaler()
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
    lambda_reg = 1e-3  # For the weights of the networks
    epoch = 100
    initial = int(0.33 * epoch)
    learning_rate = 1e-3
    clipping_value = 1

    train_loader = torch.utils.data.DataLoader(list(zip(x_train, y_train)), shuffle=True, batch_size=batch_size)
    model = VAE_model(d, D, H1, H2, activFun)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_reg)

    ELBO = np.zeros((epoch, 1))
    for i in range(epoch):
        # Initialize the losses
        train_loss = 0
        train_loss_num = 0
        for batch_idx, (x, y) in enumerate(train_loader):

            x,y=x.type(torch.FloatTensor),y.type(torch.FloatTensor)

            # MU_X_eval, LOG_X_eval, Z_ENC_eval, MU_Z_eval, LOG_VAR_Z_eval = model(x)
            # MU_X_eval, Z_ENC_eval, MU_Z_eval, LOG_VAR_Z_eval = model(x)
            MU_X_eval, Z_ENC_eval, MU_Z_eval = model(x)
             # Compute the regularization parameter
            # if initial == 0:
            #     r = 0
            # else:
            #     r = 1. * i / initial
            #     if r > 1.:
            #         r = 1.
            
             # The VAE loss
            # loss = model.VAE_loss(x=x, mu_x=MU_X_eval, log_var_x= LOG_X_eval, mu_z=MU_Z_eval, log_var_z=LOG_VAR_Z_eval, r=r)
            # loss = model.VAE_loss(x=x, mu_x=MU_X_eval, mu_z=MU_Z_eval, log_var_z=LOG_VAR_Z_eval, r=r)
            loss = model.VAE_loss(x=x, mu_x=MU_X_eval, r=1, scaler = scaler)

            # Update the parameters
            optimizer_model.zero_grad()

            # Compute the loss
            loss.backward()

            # Update the parameters
            optimizer_model.step()

             # Collect the ways
            train_loss += loss.item()
            train_loss_num += 1

        ELBO[i] = train_loss / train_loss_num
        if i % 10 == 0:
            print("[Epoch: {}/{}] [objective: {:.3f}]".format(i, epoch, ELBO[i, 0]))


    ELBO_train = ELBO[epoch-1, 0].round(2)
    print('[ELBO train: ' + str(ELBO_train) + ']')
    del MU_X_eval, MU_Z_eval, Z_ENC_eval
    # del LOG_VAR_X_eval, LOG_VAR_Z_eval
    print("Training finished")

    plt.figure()
    plt.plot(ELBO)
    plt.savefig(f'./results/train')
    plt.show()      

    torch.save(model.state_dict(),  './results/model_VE_minmax_1.pt')

if __name__ == "__main__":
    main()