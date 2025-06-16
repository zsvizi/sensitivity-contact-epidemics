import numpy as np

from src.model.model_base import EpidemicModelBase


class ValidationModel(EpidemicModelBase):
    def __init__(self, model_data) -> None:
        compartments = ["s", "e", "i", "r", "d", "inf"]

        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        iv["e"][1] = 1
        iv.update({"inf": iv["i"] + iv["r"] + iv["d"]
                   })

        iv.update({"s": self.population - (iv["e"] + iv["inf"])})

    def get_model(self, xs: np.ndarray, t, ps: dict, cm: np.ndarray) -> np.ndarray:
        s, e, i, r, d, inf = xs.reshape(-1, self.n_age)
        transmission = ps["beta"] * np.array(i).dot(cm)  # E does not infect

        model_eq_dict = {
            "s": -transmission * s / self.population,  # S'(t)
            "e": s / self.population * transmission - e * ps["alpha"],  # E'(t)
            "i": ps["alpha"] * e - ps["gamma"] * i,  # I'(t)
            "r": ps["p_recovery"] * ps["gamma"] * i,  # R'(t)
            "d": (1 - ps["p_recovery"]) * ps["gamma"] * i,  # D'(t)
            # add compartment to store total infecteds
            "inf": ps["alpha"] * e  # C'(t)
        }
        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_infected(self, solution: np.ndarray) -> np.ndarray:
        idx_e = self.c_idx["e"]
        idx_i = self.c_idx["i"]
        return self.aggregate_by_age(solution, idx_e) + \
               self.aggregate_by_age(solution, idx_i)

    def get_epidemic_peak(self, solution: np.ndarray) -> float:
        idx_e = self.c_idx["e"]
        idx_i = self.c_idx["i"]
        return (self.aggregate_by_age(solution, idx_e) +
                self.aggregate_by_age(solution, idx_i)).max()

    def get_final_size_dead(self, solution: np.ndarray) -> float:
        state = solution[-1].reshape((1, -1))
        return self.aggregate_by_age(state, self.c_idx["d"])

