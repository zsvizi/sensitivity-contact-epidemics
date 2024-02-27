import numpy as np

from src.model.model_base import EpidemicModelBase


class SirModel(EpidemicModelBase):
    def __init__(self, model_data) -> None:
        compartments = ["s", "i", "cp", "c", "r", "d"]

        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        iv["i"][3] = 1
        keys_to_copy = ["c", "r", "d"]
        iv.update({key: iv[key] for key in keys_to_copy})
        iv.update({"s": self.population - sum(iv["i"] + iv["cp"] +
                                              iv["c"] + iv["r"] + iv["d"])})

    def get_model(self, xs: np.ndarray, _, ps: dict, cm: np.ndarray) -> np.ndarray:
        beta = ps["beta"]
        s, i, cp, c, r, d = xs.reshape(-1, self.n_age)

        transmission = beta * np.dot((i + cp), cm)

        sir_eq_dict = {
            "s": -ps["susc"] * transmission * s / self.population,  # S'(t)
            "i": ps["susc"] * (1 - ps["xi"]) * (transmission * s / self.population) -
                 ps["alpha_i"] * i,  # I'(t)
            "cp": ps["susc"] * ps["xi"] * (transmission * s / self.population) -
                  ps["alpha_p"] * cp,  # C_p'(t)
            "c":  ps["alpha_p"] * cp - ps["alpha_c"] * c,  # C'(t)
            "r": ps["alpha_i"] * i + ps["alpha_c"] * c,  # R'(t)

            "d": ps["gamma_d"] * r  # R'(t)
        }

        return self.get_array_from_dict(comp_dict=sir_eq_dict)

    def get_infected(self, solution) -> np.ndarray:
        idx = self.c_idx["i"]
        return self.aggregate_by_age(solution, idx)

    def get_recovered(self, solution) -> np.ndarray:
        idx = self.c_idx["r"]
        return self.aggregate_by_age(solution, idx)

    def get_icu_dynamics(self, solution: np.ndarray) -> np.ndarray:
        idx = self.c_idx["cp"]
        return self.aggregate_by_age(solution, idx)