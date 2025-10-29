# ---------------------
#      "
#    '':''
#   ___:____      |\/|
# ,'        `.    \  /
# |  O        \___/  |
# ~^~^~^~^~^~^~^~^~^~^
# ---------------------
# @author: τ

import math
import torch


class Problem:
    def __init__(self, case=1, equation="Poisson", domain="square", eps=0.1, v_freq=6000):
        self.case = case
        self.equation = equation
        self.domain = domain
        self.eps = eps
        self.v_freq = v_freq  # Hz

    def sol(self, x, y):
        if self.equation == "Poisson":
            if self.case == 0:
                u = torch.sin(2 * torch.pi * x) * torch.sin(2 * torch.pi * y) + torch.exp(-x - y)
            elif self.case == 1:
                u = torch.sin(2 * torch.pi * x) * torch.sin(30 * torch.pi * y)
            elif self.case == 2:
                u = torch.sin(10 * torch.pi * x) * torch.sin(10 * torch.pi * y)
            elif self.case == 3:
                u = torch.sin(25 * torch.pi * x) * torch.sin(25 * torch.pi * y)
            elif self.case == 4:
                u = torch.cos(30 * torch.pi * x) * torch.cos(30 * torch.pi * y)
            elif self.case == 5:
                u = x * y + 0.1 * torch.cos(10 * x) + 0.2 * torch.sin(20 * y) + 0.3 * torch.cos(80 * x + y)
            elif self.case == 6:
                u = (torch.sin(x) + 0.1 * torch.sin(20 * x) + torch.cos(80 * x)) * (
                    torch.sin(y) + 0.1 * torch.sin(20 * y) + torch.cos(80 * y)
                )
            elif self.case == 7:
                u = torch.cos(100 * x) * torch.cos(100 * y)
            else:
                raise NotImplementedError
        elif self.equation == "multi-scale":
            eps = self.eps
            u = (
                0.25 * (x**2 + y**2) ** 2
                + eps / (16 * torch.pi) * (x**2 + y**2) * torch.sin(2 * torch.pi * (x**2 + y**2) / eps)
                + eps**2 / (32 * torch.pi**2) * torch.cos(2 * torch.pi * (x**2 + y**2) / eps)
            )
        elif self.equation == "Helmholtz":
            k = 2 * math.pi * self.v_freq / 340
            u = torch.sin(k / math.sqrt(2) * x) * torch.sin(k / math.sqrt(2) * y)
        elif self.equation == "convection-diffusion-reaction":
            u = (x - (1 - torch.exp(100 * x) / (1 - math.exp(100)))) * (
                y - (1 - torch.exp(100 * y) / (1 - math.exp(100)))
            )
        else:
            raise NotImplementedError

        return u

    def source(self, x, y):
        sin, cos, exp, pi = torch.sin, torch.cos, torch.exp, torch.pi
        if self.equation == "Poisson":
            if self.case == 0:
                f = 8 * torch.pi**2 * torch.sin(2 * torch.pi * x) * torch.sin(2 * torch.pi * y) - 2 * torch.exp(-x - y)
            elif self.case == 1:
                f = 904 * torch.pi**2 * self.sol(x, y)
            elif self.case == 2:
                f = 200 * torch.pi**2 * self.sol(x, y)
            elif self.case == 3:
                f = 1250 * torch.pi**2 * self.sol(x, y)
            elif self.case == 4:
                f = 1800 * torch.pi**2 * self.sol(x, y)
            elif self.case == 5:
                f = 80.0 * sin(20 * y) + 10.0 * cos(10 * x) + 1920.3 * cos(80 * x + y)
            elif self.case == 6:
                f = (sin(x) + 0.1 * sin(20 * x) + cos(80 * x)) * (sin(y) + 40.0 * sin(20 * y) + 6400 * cos(80 * y)) + (
                    sin(x) + 40.0 * sin(20 * x) + 6400 * cos(80 * x)
                ) * (sin(y) + 0.1 * sin(20 * y) + cos(80 * y))
            elif self.case == 7:
                f = 20000 * self.sol(x, y)
        elif self.equation == "multi-scale":
            f = -(x**2 + y**2)
        elif self.equation == "Helmholtz":
            f = 0 * x
        elif self.equation == "convection-diffusion-reaction":
            f = (
                (100 * exp(100 * x) / (1 - math.exp(100)) + 1) * (y + exp(100 * y) / (1 - math.exp(100)) - 1)
                + (100 * exp(100 * y) / (1 - math.exp(100)) + 1) * (x + exp(100 * x) / (1 - math.exp(100)) - 1)
                + 0.0001 * (x + exp(100 * x) / (1 - math.exp(100)) - 1) * (y + exp(100 * y) / (1 - math.exp(100)) - 1)
                - 100.0 * (x + exp(100 * x) / (1 - math.exp(100)) - 1) * exp(100 * y) / (1 - math.exp(100))
                - 100.0 * (y + exp(100 * y) / (1 - math.exp(100)) - 1) * exp(100 * x) / (1 - math.exp(100))
            )
        else:
            raise NotImplementedError

        return f

    def diffusion_coeff(self, x, y):
        eps = self.eps
        pi, sin, cos = torch.pi, torch.sin, torch.cos
        A = 1 / (4 + torch.cos(2 * torch.pi * (x**2 + y**2) / eps))
        A_x = 4 * pi * x * sin(2 * pi * (x**2 + y**2) / eps) / (eps * (cos(2 * pi * (x**2 + y**2) / eps) + 4) ** 2)
        A_y = 4 * pi * y * sin(2 * pi * (x**2 + y**2) / eps) / (eps * (cos(2 * pi * (x**2 + y**2) / eps) + 4) ** 2)

        return A, A_x, A_y


if __name__ == "__main__":
    from sympy import *
    from sympy.abc import x, y, pi

    eps = symbols("eps")
    u = (
        0.25 * (x**2 + y**2) ** 2
        + eps / (16 * pi) * (x**2 + y**2) * sin(2 * pi * (x**2 + y**2) / eps)
        + eps**2 / (32 * pi**2) * cos(2 * pi * (x**2 + y**2) / eps)
    )
    A = 1 / (4 + cos(2 * pi * (x**2 + y**2) / eps))

    u_x = diff(u, x)
    u_y = diff(u, y)
    u_xx, u_yy = diff(u, x, x), diff(u, y, y)
    A_x = diff(A, x)
    A_y = diff(A, y)
    print(A_y)
