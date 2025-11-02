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
import numpy as np


class Dataset:
    def __init__(self, domain: str = "square", device="cpu"):
        self.domain = domain
        self.device = device

    def collocation_points(self, nx, ny):
        if self.domain == "square":
            x, y = torch.meshgrid(torch.linspace(0, 1, nx), torch.linspace(0, 1, ny), indexing="ij")
            x, y = x.flatten(), y.flatten()
            collocation = torch.stack((x, y), dim=1)

            t = torch.linspace(0, 1, 150)
            bc = torch.cat(
                (
                    torch.stack((torch.zeros_like(t), t), dim=1),
                    torch.stack((torch.ones_like(t), t), dim=1),
                    torch.stack((t, torch.zeros_like(t)), dim=1),
                    torch.stack((t, torch.ones_like(t)), dim=1),
                ),
                dim=0,
            )
        elif self.domain == "L-shape":
            x, y = torch.meshgrid(
                torch.linspace(0, 1, nx),
                torch.linspace(0, 1, ny),
                indexing="ij",
            )
            x, y = x.flatten(), y.flatten()
            collocation = torch.stack((x, y), dim=1)
            collocation = collocation[~((x > 0.5) & (y > 0.5))]
            t1, t2 = torch.linspace(0, 1, 150), torch.linspace(0, 0.5, 70)
            bc = torch.cat(
                (
                    torch.stack((torch.zeros_like(t1), t1), dim=1),
                    torch.stack((t2, torch.ones_like(t2)), dim=1),
                    torch.stack((0.5 * torch.ones_like(t2), t2 + 0.5), dim=1),
                    torch.stack((t2 + 0.5, 0.5 * torch.ones_like(t2)), dim=1),
                    torch.stack((torch.ones_like(t2), t2), dim=1),
                    torch.stack((t1, torch.zeros_like(t1)), dim=1),
                ),
                dim=0,
            )
        elif self.domain == "circle":
            x, y = torch.meshgrid(
                torch.linspace(-0.5, 0.5, nx),
                torch.linspace(-0.5, 0.5, ny),
                indexing="ij",
            )
            x, y = x.flatten(), y.flatten()
            collocation = torch.stack((x, y), dim=1)
            t = torch.linspace(0, 2 * torch.pi, 500)
            collocation = collocation[x**2 + y**2 < 0.25]

            x_bc = 0.5 * torch.cos(t)
            y_bc = 0.5 * torch.sin(t)
            bc = torch.stack((x_bc, y_bc), dim=1)
        elif self.domain == "flower":
            x, y = torch.meshgrid(
                torch.linspace(-0.6, 0.6, nx),
                torch.linspace(-0.6, 0.6, ny),
                indexing="ij",
            )
            x, y = x.flatten(), y.flatten()
            collocation = torch.stack((x, y), dim=1)

            r = torch.sqrt(x**2 + y**2)
            cos_vec = x / torch.sqrt(x**2 + y**2)
            theta = torch.arccos(cos_vec)
            theta[y < 0] = theta[y < 0] + torch.pi
            idx = r - 0.5 + 0.1 * torch.cos(6 * theta) < 0
            collocation = collocation[idx]

            t = torch.linspace(0, 2 * torch.pi, 500)
            x_bc = (0.5 - 0.1 * torch.cos(6 * t)) * torch.cos(t)
            y_bc = (0.5 - 0.1 * torch.cos(6 * t)) * torch.sin(t)
            bc = torch.stack((x_bc, y_bc), dim=1)
        elif self.domain == "kite":
            x, y = torch.meshgrid(
                torch.linspace(-0.6, 0.6, nx),
                torch.linspace(-0.6, 0.6, ny),
                indexing="ij",
            )
            x, y = x.flatten(), y.flatten()
            collocation = torch.stack((x, y), dim=1)
            rho = (x + 0.2 - 0.3 * (1 - 2 * (y / 0.6) ** 2)) / 0.5
            idx = rho**2 + (y / 0.6) ** 2 - 1 < 0
            collocation = collocation[idx]

            t = torch.linspace(0, 2 * torch.pi, 500)
            x_bc = 0.5 * torch.cos(t) + 0.3 * torch.cos(2 * t) - 0.2
            y_bc = 0.6 * torch.sin(t)
            bc = torch.stack((x_bc, y_bc), dim=1)
        else:
            raise NotImplementedError

        return collocation.to(self.device), bc.to(self.device)

    def test_points(self):
        if self.domain == "square" or self.domain == "space-time":
            x_test, _ = self.collocation_points(120, 120)
        else:
            mesh = np.load(f"./test_points/{self.domain}.npz")
            x_test = torch.from_numpy(mesh["vertices"])

        return x_test.to(self.device)
