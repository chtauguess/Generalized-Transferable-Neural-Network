# ---------------------
#      "
#    '':''
#   ___:____      |\/|
# ,'        `.    \  /
# |  O        \___/  |
# ~^~^~^~^~^~^~^~^~^~^
# ---------------------
# @author: τ

import torch
import torch.nn as nn


class GTransNet(nn.Module):
    def __init__(self, hidden_dim=600, output_dim=500, gamma=2):
        super().__init__()

        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)

        self.fc1.weight.data = torch.randn(hidden_dim, 2)
        self.fc1.bias.data = torch.randn(hidden_dim)
        self.fc1.weight.data /= torch.sqrt(torch.sum(self.fc1.weight.data**2, dim=1, keepdim=True))

        a = torch.sqrt(torch.tensor(0.8) / (hidden_dim))
        self.fc2.weight.data = a * torch.randn(output_dim, hidden_dim)
        self.gamma = gamma
        self.u1, self.u2 = None, None

    def forward(self, X):
        self.u1 = torch.tanh(self.gamma * self.fc1(X))
        self.u2 = self.fc2(self.u1)
        u = torch.tanh(self.u2)
        return u

    def derivative(self, u, order=1):
        d_u1, d_u = 1 - self.u1**2, 1 - u**2
        u1_x = self.gamma * self.fc1.weight.T[[0], :] * d_u1
        u1_y = self.gamma * self.fc1.weight.T[[1], :] * d_u1
        u2_x, u2_y = self.fc2(u1_x), self.fc2(u1_y)
        u_x, u_y = d_u * u2_x, d_u * u2_y
        if order == 2:
            d2_u1 = 2 * self.u1 * d_u1
            u1_xx = -((self.gamma * self.fc1.weight.T.data[[0], :]) ** 2) * d2_u1
            u1_yy = -((self.gamma * self.fc1.weight.T.data[[1], :]) ** 2) * d2_u1
            u1_xy = -(self.gamma**2) * self.fc1.weight.data.T[[0], :] * self.fc1.weight.data.T[[1], :] * d2_u1
            u2_xx, u2_yy, u2_xy = self.fc2(u1_xx), self.fc2(u1_yy), self.fc2(u1_xy)
            u_xx = -2 * u * u_x * u2_x + d_u * u2_xx
            u_yy = -2 * u * u_y * u2_y + d_u * u2_yy
            u_xy = -2 * u * u_y * u2_x + d_u * u2_xy
            return u_x, u_y, u_xx, u_yy, u_xy

        return u_x, u_y


class TransNet(nn.Module):
    def __init__(self, output_dim, gamma=2):
        super().__init__()
        self.fc = nn.Linear(2, output_dim)

        self.fc.weight.data = torch.randn(output_dim, 2)
        self.fc.bias.data = torch.rand(output_dim)
        self.fc.weight.data /= torch.sqrt(torch.sum(self.fc.weight.data**2, dim=1, keepdim=True))
        self.gamma = gamma

    def forward(self, X):
        return torch.tanh(self.gamma * self.fc(X))

    def derivative(self, u, order=1):
        d_u = 1 - u**2
        u_x, u_y = (
            self.gamma * d_u * self.fc.weight.T[[0], :],
            self.gamma * d_u * self.fc.weight.T[[1], :],
        )
        if order == 2:
            d2_u = 2 * u * d_u
            u_xx, u_yy = (
                -((self.gamma * self.fc.weight.T.data[[0], :]) ** 2) * d2_u,
                -((self.gamma * self.fc.weight.T.data[[1], :]) ** 2) * d2_u,
            )
            u_xy = -(self.gamma**2) * self.fc.weight.data.T[[0], :] * self.fc.weight.data.T[[1], :] * d2_u
            return u_x, u_y, u_xx, u_yy, u_xy

        return u_x, u_y


class ELM(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.fc = nn.Linear(2, output_dim)

        self.fc.weight.data = 6 * torch.rand(output_dim, 2) - 3
        self.fc.bias.data = 6 * torch.rand(output_dim) - 3

    def forward(self, X):
        return torch.tanh(self.fc(X))

    def derivative(self, u, order=1):
        d_u = 1 - u**2
        u_x, u_y = (
            d_u * self.fc.weight.T[[0], :],
            d_u * self.fc.weight.T[[1], :],
        )
        if order == 2:
            d2_u = 2 * u * d_u
            u_xx, u_yy = (
                -((self.fc.weight.T.data[[0], :]) ** 2) * d2_u,
                -((self.fc.weight.T.data[[1], :]) ** 2) * d2_u,
            )
            u_xy = -self.fc.weight.data.T[[0], :] * self.fc.weight.data.T[[1], :] * d2_u
            return u_x, u_y, u_xx, u_yy, u_xy

        return u_x, u_y
