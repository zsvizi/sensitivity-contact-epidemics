import numpy as np

from src.model.model_base import EpidemicModelBase


class SirModel(EpidemicModelBase):
    def __init__(self, model_data) -> None:
        self.compartments = ["s", "i", "r"]

        super().__init__(model_data=model_data, compartments=self.compartments)

    def update_initial_values(self, iv: dict):
        iv = {key: np.zeros(self.n_age) for key in self.compartments}
        iv["i"][3] = 1
        iv.update({"r": iv["r"]})

        iv.update({"s": self.population - (iv["i"] + iv["r"])
                   })

    def get_model(self, xs: np.ndarray, _, ps: dict, cm: np.ndarray) -> np.ndarray:
        susc = np.array(ps["susc"])
        gamma = ps["gamma_a"]  # gamma_a = 0.333
        beta = ps["beta"]

        s, i, r = xs.reshape(-1, self.n_age)

        transmission = beta * np.array(i).dot(cm)

        sir_eq_dict = {
            "s": -susc * (s / self.population) * transmission,  # S'(t)
            "i": susc * (s / self.population) * transmission - gamma * i,  # I'(t)
            "r": gamma * i  # R'(t)
        }

        return self.get_array_from_dict(comp_dict=sir_eq_dict)

    def get_infected(self, solution) -> np.ndarray:
        idx = self.c_idx["i"]
        return self.aggregate_by_age(solution, idx)

    def get_recovered(self, solution) -> np.ndarray:
        idx = self.c_idx["r"]
        return self.aggregate_by_age(solution, idx)
