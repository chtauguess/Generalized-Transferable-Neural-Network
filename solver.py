# ---------------------
#      "
#    '':''
#   ___:____      |\/|
# ,'        `.    \  /
# |  O        \___/  |
# ~^~^~^~^~^~^~^~^~^~^
# ---------------------
# @author: τ

import time
import torch
import argparse

from problem import Problem
from dataset import Dataset
from method import GTNN, TransNet, ELM


class Solver:
    def __init__(self, args, seed=0):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(seed)
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.output_dim = args.output_dim
        self.dataset = Dataset(args.domain, self.device)
        self.train_data, self.bc = self.dataset.collocation_points(args.nx, args.ny)

        self.prob = Problem(args.case, args.equation, args.domain, args.eps, args.v_freq)
        if args.model == "GTNN":
            self.model = GTNN(args.hidden_dim, args.output_dim, args.gamma).to(self.device)
        elif args.model == "TransNet":
            self.model = TransNet(args.output_dim, args.gamma).to(self.device)
        elif args.model == "ELM":
            self.model = ELM(args.output_dim).to(self.device)
        else:
            raise NotImplementedError

    def solve(self, equation):
        tt = time.time()
        print(f"{6 * '=='} start (domain: {self.args.domain}, gamma: {self.args.gamma}) {6 * '=='}")
        print(f"collocation: {self.train_data.shape[0]}, bc: {self.bc.shape[0]}")
        print(f"features shape = {self.train_data.shape[0] + self.bc.shape[0], self.output_dim}")
        u_pde = self.model(self.train_data)
        u_x, u_y, u_xx, u_yy, _ = self.model.derivative(u_pde, order=2)
        u_bc = self.model(self.bc)
        if equation == "Poisson":
            feature_pde, feature_bc = -u_xx - u_yy, u_bc
        elif equation == "Helmholtz":  # domain: kite or flower
            k = 2 * torch.pi * self.args.v_freq / 340
            feature_pde, feature_bc = -u_xx - u_yy - k**2 * u_pde, u_bc
        elif equation == "multi-scale":
            A, A_x, A_y = self.prob.diffusion_coeff(self.train_data[:, [0]], self.train_data[:, [1]])
            feature_pde = -(A_x * u_x + A_y * u_y + A * (u_xx + u_yy))
            feature_bc = u_bc
        elif equation == "convection-diffusion-reaction":
            # -k \delta u + a \cdot \nable u + su = f (k=0.01,a=(1,1),s = 10^{-4})
            # domain: square (strong boundary layer problem)
            k, s = 0.01, 10**-4
            feature_pde, feature_bc = -k * (u_xx + u_yy) + (u_x + u_y) + s * u_pde, u_bc
        else:
            raise NotImplementedError

        target_pde = self.prob.source(self.train_data[:, 0], self.train_data[:, 1])
        target_bc = self.prob.sol(self.bc[:, 0], self.bc[:, 1])
        eta_pde = 1 / max(torch.max(torch.abs(feature_pde)), torch.max(torch.abs(target_pde)))
        eta_bc = 1 / max(torch.max(torch.abs(feature_bc)), torch.max(torch.abs(target_bc)))
        features = torch.vstack([eta_pde * feature_pde, eta_bc * feature_bc])
        targets = torch.concatenate([eta_pde * target_pde, eta_bc * target_bc])

        tt1 = time.time()
        print(f"Assembling: {(tt1 - tt):.2f}s")
        w = torch.linalg.lstsq(features, targets)[0]
        print(f"Solving: {(time.time() - tt1):.2f}s")
        print(f"{6 * '=='} end (elapsed {(time.time() - tt):.2f}s) {6 * '=='}")
        return w

    def test(self, equation):
        w = self.solve(equation)
        x_test, triangles = self.dataset.test_points()
        u_pde = self.model(x_test)
        u_pred = u_pde @ w
        u_true = self.prob.sol(x_test[:, 0], x_test[:, 1])
        abs_err = torch.abs(u_pred - u_true)
        r_l2_err = torch.sqrt(torch.linalg.norm(abs_err**2) / torch.linalg.norm(u_true**2))
        r_max_err = torch.max(abs_err) / torch.max(u_true)
        print(
            f"test size: {x_test.shape[0]},",
            f"max err: {r_max_err:.2e}, L2 err: {r_l2_err:.2e}",
        )
        abs_err = abs_err.detach().cpu().numpy()
        x_test = x_test.detach().cpu().numpy()
        u_pred = u_pred.detach().cpu().numpy()

        return x_test, triangles, u_pred, r_max_err.detach(), r_l2_err.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=120)
    parser.add_argument("--ny", type=int, default=120)
    parser.add_argument("--hidden_dim", type=int, default=6000)
    parser.add_argument("--output_dim", type=int, default=4000)
    parser.add_argument("--gamma", type=float, default=8)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--eps", type=float, default=0.1, help="multi-scale probelm only")
    parser.add_argument("--v_freq", type=float, default=6000, help="Helmholtz probelm only")
    parser.add_argument(
        "--equation",
        type=str,
        default="Poisson",
        help="Poisson, multi-scale, Helmholtz or convection-diffusion-reaction",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GTNN",
        help="GTNN, TransNet or ELM",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="circle",
        help="square, L-shape, circle, kite or flower",
    )
    parser.add_argument("--save", type=bool, default=False)
    args = parser.parse_args()

    solver = Solver(args)
    x_test, triangles, u_pred, r_max_err, r_L2_err = solver.test(args.equation)
