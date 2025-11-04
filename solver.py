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
from models import GTransNet, TransNet


class Solver:
    def __init__(self, args, seed=0):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(seed)
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.output_dim = args.output_dim
        self.dataset = Dataset(args.domain, self.device)
        self.train_data, self.bc = self.dataset.collocation_points(args.nx, args.ny)

        self.prob = Problem(args.cases, args.equation, args.domain, args.eps, args.v_freq)
        print(f"model: {args.model}")
        if args.model == "GTransNet":
            self.model = GTransNet(args.hidden_dim, args.output_dim, args.gamma).to(self.device)
        elif args.model == "TransNet":
            self.model = TransNet(args.output_dim, args.gamma).to(self.device)
        else:
            raise NotImplementedError

    def solve(self, equation):
        tt = time.time()
        print(
            f"{6 * '=='} start (equation: {equation}, domain: {self.args.domain}, gamma: {self.args.gamma}) {6 * '=='}"
        )
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

    def test(self, w):
        x_test = self.dataset.test_points()
        u_pde = self.model(x_test)
        u_pred = u_pde @ w
        u_true = self.prob.sol(x_test[:, 0], x_test[:, 1])
        abs_err = torch.abs(u_pred - u_true)
        r_l2_err = torch.sqrt(torch.linalg.norm(abs_err**2) / torch.linalg.norm(u_true**2))
        r_max_err = torch.max(abs_err) / torch.max(torch.abs(u_true))
        print(
            f"test size: {x_test.shape[0]},",
            f"max err: {r_max_err:.2e}, L2 err: {r_l2_err:.2e}",
        )

        return r_max_err.detach(), r_l2_err.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GTransNet", help="GTransNet or TransNet")
    parser.add_argument("--test", type=str, default="test0", help="test0-test10")
    args = parser.parse_args()

    from dataclasses import dataclass

    @dataclass
    class Setting:
        nx: int
        ny: int
        hidden_dim: int
        output_dim: int
        gamma: float = 8
        cases: int = 1
        eps: float = 0.1
        v_freq: float = 6000
        equation: str = "Poisson"  # Poisson, multi-scale, Helmholtz or convection-diffusion-reaction
        domain: str = "square"  # square, L-shape, circle, kite, flower
        model: str = "GTransNet"  # GTransNet, TransNet or ELM

    test_setting = {
        "test0": Setting(30, 30, 800, 800, gamma=2, cases=0, domain="square", model=args.model),
        "test1": Setting(120, 120, 6000, 4000, cases=1, domain="kite", model=args.model),
        "test2": Setting(120, 120, 6000, 4000, cases=2, domain="circle", model=args.model),
        "test3": Setting(120, 120, 6000, 4000, cases=3, gamma=10, domain="L-shape", model=args.model),
        "test4": Setting(140, 140, 6000, 4000, v_freq=4000, equation="Helmholtz", domain="flower", model=args.model),
        "test5": Setting(
            140, 140, 6000, 4000, v_freq=6000, gamma=10, equation="Helmholtz", domain="flower", model=args.model
        ),
        "test6": Setting(
            140, 140, 6000, 4000, v_freq=8000, gamma=10, equation="Helmholtz", domain="flower", model=args.model
        ),
        "test7": Setting(100, 100, 6000, 4000, eps=0.5, equation="multi-scale", model=args.model),
        "test8": Setting(100, 100, 6000, 4000, eps=0.2, equation="multi-scale", model=args.model),
        "test9": Setting(100, 100, 6000, 4000, eps=0.1, equation="multi-scale", model=args.model),
        "test10": Setting(100, 100, 6000, 4000, equation="convection-diffusion-reaction", model=args.model),
    }

    solver = Solver(test_setting[args.test])
    w = solver.solve(test_setting[args.test].equation)
    r_max_err, r_L2_err = solver.test(w)

