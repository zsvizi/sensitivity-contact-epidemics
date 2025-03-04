import math

import numpy as np

from src.model.model_base import EpidemicModelBase


class SeirUK(EpidemicModelBase):
    def __init__(self, model_data) -> None:
        compartments = ["s", "e", "i", "r", "c"]

        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        iv["i"][3] = 1
        iv.update({"e": iv["e"], "r": iv["r"]})

        iv.update({"s": self.population - (iv["e"] + iv["i"] + iv["r"] + iv["c"])})

    def get_model(self, xs: np.ndarray, t, ps: dict, cm: np.ndarray) -> np.ndarray:
        # the same order as in self.compartments!
        s, e, i, r, c = xs.reshape(-1, self.n_age)
        transmission = ps["beta"] * np.array(i).dot(cm)
        # add seasonality
        z = 1 + ps["q"] * np.sin(2 * math.pi * (t - ps["t_offset"]) / 365)
        transmission *= z
        model_eq_dict = {
            "s": -transmission * s / self.population,  # S'(t)
            "e": s / self.population * transmission - e * ps["gamma"],  # E'(t)
            "i": ps["gamma"] * e - ps["rho"] * i,  # I'(t)
            "r": ps["rho"] * i,  # R'(t)

            # add compartment to store total infecteds
            "c": s / self.population * transmission + ps["gamma"] * e  # C'(t)
        }
        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_infected(self, solution) -> np.ndarray:
        idx = self.c_idx["i"]
        return self.aggregate_by_age(solution, idx)

    def get_recovered(self, solution) -> np.ndarray:
        idx = self.c_idx["r"]
        return self.aggregate_by_age(solution, idx)
