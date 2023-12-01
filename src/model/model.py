import numpy as np

from src.model.model_base import EpidemicModelBase


class RostModelHungary(EpidemicModelBase):
    def __init__(self, model_data, country: str = "Hungary") -> None:
        self.country = country
        compartments = ["s", "l1", "l2",
                        "ip", "ia1", "ia2", "ia3",
                        "is1", "is2", "is3",
                        "ih", "ic", "icr",
                        "r", "d", "c"]
        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        iv["l1"][2] = 1
        iv.update({"c": iv["ip"] + iv["ia1"] + iv["ia2"] + iv["ia3"] + iv["is1"] +
                   iv["is2"] + iv["is3"] + iv["r"] + iv["d"]
                   })

        iv.update({"s": self.population - (iv["c"] + iv["l1"] + iv["l2"])})

    def get_model(self, xs: np.ndarray, _, ps: dict, cm: np.ndarray) -> np.ndarray:
        # the same order as in self.compartments!
        s, l1, l2, ip, ia1, ia2, ia3, is1, is2, is3, ih, ic, icr, r, d, c = xs.reshape(-1, self.n_age)

        transmission = ps["beta"] * np.array((ip + ps["inf_a"] * (ia1 + ia2 + ia3) + (is1 + is2 + is3))).dot(cm)
        actual_population = self.population

        model_eq_dict = {
            "s": -ps["susc"] * (s / actual_population) * transmission,  # S'(t)
            "l1": ps["susc"] * (s / actual_population) * transmission - 2 * ps["alpha_l"] * l1,  # L1'(t)
            "l2": 2 * ps["alpha_l"] * l1 - 2 * ps["alpha_l"] * l2,  # L2'(t)

            "ip": 2 * ps["alpha_l"] * l2 - ps["alpha_p"] * ip,  # Ip'(t)

            "ia1": ps["p"] * ps["alpha_p"] * ip - 3 * ps["gamma_a"] * ia1,  # Ia1'(t)
            "ia2": 3 * ps["gamma_a"] * ia1 - 3 * ps["gamma_a"] * ia2,  # Ia2'(t)
            "ia3": 3 * ps["gamma_a"] * ia2 - 3 * ps["gamma_a"] * ia3,  # Ia3'(t)

            "is1": (1 - ps["p"]) * ps["alpha_p"] * ip - 3 * ps["gamma_s"] * is1,  # Is1'(t)
            "is2": 3 * ps["gamma_s"] * is1 - 3 * ps["gamma_s"] * is2,  # Is2'(t)
            "is3": 3 * ps["gamma_s"] * is2 - 3 * ps["gamma_s"] * is3,  # Is3'(t)
            "ih": ps["h"] * (1 - ps["xi"]) * 3 * ps["gamma_s"] * is3 - ps["gamma_h"] * ih,  # Ih'(t)
            "ic": ps["h"] * ps["xi"] * 3 * ps["gamma_s"] * is3 - ps["gamma_c"] * ic,  # Ic'(t)
            "icr": (1 - ps["mu"]) * ps["gamma_c"] * ic - ps["gamma_cr"] * icr,  # Icr'(t)

            "r": 3 * ps["gamma_a"] * ia3 + (1 - ps["h"]) * 3 * ps["gamma_s"] * is3 + ps["gamma_h"] * ih +
            ps["gamma_cr"] * icr,  # R'(t)
            "d": ps["mu"] * ps["gamma_c"] * ic,  # D'(t)

            "c": 2 * ps["alpha_l"] * l2  # C'(t)
        }
        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_hospitalized(self, solution: np.ndarray) -> np.ndarray:
        idx = self.c_idx["ih"]
        idx_2 = self.c_idx["icr"]
        return self.aggregate_by_age(solution, idx) + self.aggregate_by_age(solution, idx_2)

    def get_ventilated(self, solution: np.ndarray) -> np.ndarray:
        idx = self.c_idx["ic"]
        return self.aggregate_by_age(solution, idx)
