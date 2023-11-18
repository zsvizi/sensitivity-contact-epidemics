import numpy as np
import torch
from torch import Tensor


class EquationGenerator:
    def __init__(self, ps, actual_population) -> None:
        self.ps = ps
        self.population = actual_population
        self.n_age = self.population.shape[0]

    def evaluate_equations(self, _, cm, xs: np.ndarray):
        ps = self.ps
        # the same order as in self.compartments!
        s, l1, l2, ip, ia1, ia2, ia3, is1, is2, is3, ih, ic, icr, r, d, c = xs.reshape(-1, self.n_age)

        transmission = ps["beta"] * np.array((ip + ps["inf_a"] * (ia1 + ia2 + ia3) + (is1 + is2 + is3))).dot(cm)
        actual_population = self.population

        model_eq = [
            torch.Tensor(-ps["susc"]) * (s / actual_population) * transmission,  # S'(t)
            torch.Tensor(ps["susc"]) * (s / actual_population) * transmission - 2 * ps["alpha_l"] * l1,  # L1'(t)
            2 * ps["alpha_l"] * l1 - 2 * ps["alpha_l"] * l2,  # L2'(t)

            2 * ps["alpha_l"] * l2 - ps["alpha_p"] * ip,  # Ip'(t)

            torch.Tensor(ps["p"]) * ps["alpha_p"] * ip - 3 * ps["gamma_a"] * ia1,  # Ia1'(t)
            3 * ps["gamma_a"] * ia1 - 3 * ps["gamma_a"] * ia2,  # Ia2'(t)
            3 * ps["gamma_a"] * ia2 - 3 * ps["gamma_a"] * ia3,  # Ia3'(t)

            (1 - torch.Tensor(ps["p"])) * ps["alpha_p"] * ip - 3 * ps["gamma_s"] * is1,  # Is1'(t)
            3 * ps["gamma_s"] * is1 - 3 * ps["gamma_s"] * is2,  # Is2'(t)
            3 * ps["gamma_s"] * is2 - 3 * ps["gamma_s"] * is3,  # Is3'(t)

            torch.Tensor(ps["h"]) * (1 - ps["xi"]) * 3 * ps["gamma_s"] * is3 - ps["gamma_h"] * ih,  # Ih'(t)
            torch.Tensor(ps["h"]) * ps["xi"] * 3 * ps["gamma_s"] * is3 - ps["gamma_c"] * ic,  # Ic'(t)
            (1 - torch.Tensor(ps["mu"])) * ps["gamma_c"] * ic - ps["gamma_cr"] * icr,  # Icr'(t)

            3 * ps["gamma_a"] * ia3 + (1 - torch.Tensor(ps["h"])) *
            3 * ps["gamma_s"] * is3 + ps["gamma_h"] * ih + ps["gamma_cr"] * icr,  # R'(t)
            torch.Tensor(ps["mu"]) * ps["gamma_c"] * ic,  # D'(t)

            2 * ps["alpha_l"] * l2  # C'(t)
        ]
        print("d", d)
        return model_eq


