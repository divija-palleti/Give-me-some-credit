
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import pairwise_distances
import gc


class VAE_model(nn.Module):

    def __init__(self, d, D, H1, H2, activFun):
        super(VAE_model, self).__init__()

        # The VAE components
        self.enc = nn.Sequential(
            nn.Linear(D, H1),
            nn.BatchNorm1d(H1),
            activFun,
            nn.Linear(H1, H2),
            nn.BatchNorm1d(H2),
            activFun
        )

        self.mu_enc = nn.Sequential(
            self.enc,
            nn.Linear(H2, d)
        )

        self.log_var_enc = nn.Sequential(
            self.enc,
            nn.Linear(H2, d)
        )

        self.dec = nn.Sequential(
            nn.Linear(d, H2),
            nn.BatchNorm1d(H2),
            activFun,
            nn.Linear(H2, H1),
            nn.BatchNorm1d(H1),
            activFun
        )

        self.mu_dec = nn.Sequential(
            self.dec,
            nn.Linear(H1, D),
            nn.BatchNorm1d(D),
            nn.Sigmoid() # only for minmax
        )

        self.log_var_dec = nn.Sequential(
            self.dec,
            nn.Linear(H1, D)
        )

    def encode(self, x):
        return self.mu_enc(x)
        # return self.mu_enc(x), self.log_var_enc(x)

    def decode(self, z):

        return self.mu_dec(z)

    @staticmethod
    def reparametrization_trick(mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  # the Gaussian random noise
        return mu + std * epsilon

    def forward(self, x):

        mu_z = self.encode(x)
        # mu_z, log_var_z = self.encode(x)
        # z_rep = self.reparametrization_trick(mu_z, log_var_z)
        z_rep = mu_z
        mu_x = self.decode(z_rep)
        return mu_x, z_rep, mu_z

        # return mu_x, z_rep, mu_z, log_var_z

    def predict(self, data):
        return self.forward(data)
    
    def regenerate(self, z, grad=False):
        mu_x, log_var_x = self.decode(z)
        return mu_x


    # # Computes the objective function of the VAE
    # def VAE_loss(self, x, mu_x, log_var_x, mu_z, log_var_z, r=1.0):
    #     D = mu_x.shape[1]
    #     d = mu_z.shape[1]


    #     if log_var_x.shape[1] == 1:
    #         P_X_Z = + 0.5 * (D * log_var_x + (((x - mu_x) ** 2) / log_var_x.exp()).sum(dim=1, keepdim=True)).mean()
    #     else:
    #         P_X_Z = + 0.5 * (log_var_x.sum(dim=1, keepdim=True)
    #                         + (((x - mu_x) ** 2) / log_var_x.exp()).sum(dim=1, keepdim=True)).mean()

    #     if log_var_z.shape[1] == 1:
    #         Q_Z_X = - 0.5 * (d * log_var_z).mean()
    #     else:
    #         Q_Z_X = - 0.5 * log_var_z.sum(dim=1, keepdim=True).mean()

    #     if log_var_z.shape[1] == 1:
    #         P_Z = + 0.5 * ((mu_z ** 2).sum(dim=1, keepdim=True) + d * log_var_z.exp()).mean()
    #     else:
    #         P_Z = + 0.5 * ((mu_z ** 2).sum(dim=1, keepdim=True) + log_var_z.exp().sum(dim=1, keepdim=True)).mean()

    #     return P_X_Z + r * Q_Z_X + r * P_Z

        










    def VAE_loss(self, x, mu_x, r, scaler, mu_z=0, log_var_z=0 ):
    # def VAE_loss(self, x, mu_x, r = 1.0):


        # reconstructionLoss = 0.5 * torch.sum(torch.sqrt((x  - mu_x)**2)) # sum of square root of input from reconstructed 
        # KL = -0.5 * torch.sum(1+ log_var_z - mu_z.pow(2) - log_var_z.exp())
        # total = torch.mean(KL + reconstructionLoss)
        reconstructionLoss =  torch.sum(((x  - mu_x)**2))
        total = torch.mean(reconstructionLoss) 
        # x_1 = x.detach().numpy()
        # x_1 = scaler.inverse_transform(x_1)
        # x_1 = torch.from_numpy(x_1)
        # x_1 = x_1.requires_grad_(True)
        # x_2 = mu_x.detach().numpy()
        # x_2 = scaler.inverse_transform(x_2)
        # x_2 = torch.from_numpy(x_2)
        # x_2 = x_2.requires_grad_(True)

       
        # reconstructionLoss_1 =  torch.sum(((x_1  - x_2)**2)) 
        # reconstructionLoss =  torch.abs((x  - mu_x))# sum of square root of input from reconstructed 
        # total = torch.mean(reconstructionLoss)
        # total_1 = torch.mean(reconstructionLoss_1)

        # print(total, "jjkj")
        # print(total_1, "1111")
        # print(jjjjjjj)

	    # KL = -0.5 * torch.sum(1+ log_var_z - mu_z.pow(2) - log_var_z.exp())  # KL divergence 
		# total = torch.mean(KL + reconstructionLoss)   # 
        # print(mu_x)
        # reproduction_loss = nn.functional.binary_cross_entropy(mu_x, x, reduction='mean')
        # print(reproduction_loss)
    #     kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)
    # # loss = recon_loss + kl_loss
    #     KLD      = - 0.5 * torch.sum(1+ log_var_z - mu_z.pow(2) - log_var_z.exp())
    #     print(KLD)
    #     print(reproduction_loss + 1 * KLD)
        # print(kkk)

        return total

        