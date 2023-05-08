from abc import ABC, abstractmethod

import numpy as np
import torch
from torchdiffeq import odeint as ode
from scipy.integrate import odeint
from src.model.model_torch import EpidemicModel, Epidemic


class EpidemicModelBase(ABC):
    def __init__(self, model_data, compartments: list, run_ode: str = "tor") -> None:
        self.population = model_data.age_data.flatten()
        self.compartments = compartments
        self.c_idx = {comp: idx for idx, comp in enumerate(self.compartments)}
        self.n_age = self.population.shape[0]
        self.run_ode = run_ode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize(self):
        iv = {key: np.zeros(self.n_age) for key in self.compartments}
        return iv

    def aggregate_by_age(self, solution, idx) -> np.ndarray:
        return np.sum(solution[:, idx * self.n_age:(idx + 1) * self.n_age], axis=1)

    def get_cumulative(self, solution) -> np.ndarray:
        idx = self.c_idx["c"]
        return self.aggregate_by_age(solution, idx)

    def get_deaths(self, solution) -> np.ndarray:
        idx = self.c_idx["d"]
        return self.aggregate_by_age(solution, idx)

    def get_solution(self, t: int, parameters: dict, cm: np.ndarray):
        initial_values = self.get_initial_values()

        solution = EpidemicModel(self, population=self.population, cm=cm, ps=parameters).to(self.device)
        iv_torch = Epidemic(self.population, cm=cm, params=parameters)
        if self.run_ode == "torch":
            return ode(solution.forward, iv_torch.get_initial_values(), t)
        else:
            return np.array(odeint(self.get_model, initial_values, t, args=(parameters, cm)))

    def get_array_from_dict(self, comp_dict) -> np.ndarray:
        return np.array([comp_dict[comp] for comp in self.compartments]).flatten()

    def get_initial_values(self) -> np.ndarray:
        iv = self.initialize()
        self.update_initial_values(iv=iv)
        return self.get_array_from_dict(comp_dict=iv)

    @abstractmethod
    def update_initial_values(self, iv: dict):
        pass

    @abstractmethod
    def get_model(self, xs, ts, ps, cm):
        pass