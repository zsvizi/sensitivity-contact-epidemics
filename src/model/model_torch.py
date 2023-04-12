import numpy as np
import torch


class Epidemic:
    def __init__(self, population: np.ndarray, params: dict, cm: np.ndarray):
        self.population = population
        self.params = params
        self.cm = cm
        self.n_age = self.population.shape[0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_initial_values(self):
        init_values = torch.cat([torch.from_numpy(self.population), torch.zeros(self.n_age).to(self.device),
                                 torch.zeros(self.n_age).to(self.device), torch.zeros(self.n_age).to(self.device),
                                 torch.zeros(self.n_age).to(self.device), torch.zeros(self.n_age).to(self.device),
                                 torch.zeros(self.n_age).to(self.device), torch.zeros(self.n_age).to(self.device),
                                 torch.zeros(self.n_age).to(self.device), torch.zeros(self.n_age).to(self.device),
                                 torch.zeros(self.n_age).to(self.device), torch.zeros(self.n_age).to(self.device),
                                 torch.zeros(self.n_age).to(self.device), torch.zeros(self.n_age).to(self.device),
                                 torch.zeros(self.n_age).to(self.device),
                                 torch.zeros(self.n_age).to(self.device)]).to(self.device)
        init_values[18] = 1
        return init_values


class EpidemicModel(torch.nn.Module):
    def __init__(self, model, ps: dict, population: np.ndarray, cm: np.ndarray):
        super(EpidemicModel, self).__init__()
        self.model = model
        self.cm = cm

        self.population = population
        self.ps = ps
        self.n_age = self.population.shape[0]

    def forward(self, _, xs: np.ndarray):
        # Set initial conditions
        ps = self.ps
        s, l1, l2, ip, ia1, ia2, ia3, is1, is2, is3, ih, ic, icr, r, d, c = xs.reshape(-1, self.n_age)

        transmission = ps["beta"] * np.array((ip + ps["inf_a"] * (ia1 + ia2 + ia3) + (is1 + is2 + is3))).dot(self.cm)
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

            3 * ps["gamma_a"] * ia3 + (1 - torch.Tensor(ps["h"])) * 3 * ps["gamma_s"] * is3 + ps["gamma_h"] * ih +
            ps["gamma_cr"] * icr,  # R'(t)
            torch.Tensor(ps["mu"]) * ps["gamma_c"] * ic,  # D'(t)

            2 * ps["alpha_l"] * l2  # C'(t)
        ]
        return torch.cat(model_eq)
