import numpy as np

from src.model.model_base import EpidemicModelBase


class SirModel(EpidemicModelBase):
    def __init__(self, model_data) -> None:
        compartments = ["s", "i", "cp", "c", "r", "d", "inf", "hosp", "icu"]
        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        iv["i"][3] = 1
        keys_to_copy = ["c", "r", "d", "inf", "hosp", "icu"]
        iv.update({key: iv[key] for key in keys_to_copy})
        iv.update({"s": self.population - sum(iv["i"] + iv["cp"] +
                                              iv["c"] + iv["r"] + iv["d"])})

    def get_model(self, xs: np.ndarray, _, ps: dict, cm: np.ndarray) -> np.ndarray:
        beta = ps["beta"]
        s, i, cp, c, r, d, inf, hosp, icu = xs.reshape(-1, self.n_age)

        transmission = beta * np.dot((i + cp), cm)

        sir_eq_dict = {
            "s": -ps["susc"] * transmission * s / self.population,  # S'(t)
            "i": ps["susc"] * (1 - ps["xi"]) * (transmission * s / self.population) -
                 ps["alpha_i"] * i,  # I'(t)
            "cp": ps["susc"] * ps["xi"] * (transmission * s / self.population) -
                  ps["alpha_p"] * cp,  # C_p'(t)
            "c":  ps["alpha_p"] * cp - ps["alpha_c"] * c,  # C'(t)
            "r": ps["alpha_i"] * i + ps["alpha_c"] * c,  # R'(t)
            "d": ps["gamma_d"] * r,  # R'(t)
            # add compartments to collect epidemics totals
            "inf": ps["susc"] * (1 - ps["xi"]) * (transmission * s / self.population) +
                   ps["susc"] * ps["xi"] * (transmission * s / self.population),  # Inf'(t)
            "hosp": ps["susc"] * ps["xi"] * (transmission * s / self.population) +
                    ps["susc"] * ps["xi"] * (transmission * s / self.population),  # Hos'(t)
            "icu": ps["alpha_p"] * cp  # ICU'(t)
        }

        return self.get_array_from_dict(comp_dict=sir_eq_dict)

    def get_infected(self, solution: np.ndarray) -> np.ndarray:
        idx_i = self.c_idx["i"]
        idx_cp = self.c_idx["cp"]
        idx_c = self.c_idx["c"]
        return (
                self.aggregate_by_age(solution, idx_i) +
                self.aggregate_by_age(solution, idx_cp) +
                self.aggregate_by_age(solution, idx_c)
        )

    def get_epidemic_peak(self, solution: np.ndarray) -> float:
        idx_i = self.c_idx["i"]
        idx_cp = self.c_idx["cp"]
        idx_c = self.c_idx["c"]
        return (
                self.aggregate_by_age(solution, idx_i) +
                self.aggregate_by_age(solution, idx_cp) +
                self.aggregate_by_age(solution, idx_c)
        ).max()

    def get_hospital_peak(self, solution: np.ndarray) -> float:
        idx_cp = self.c_idx["cp"]
        idx_c = self.c_idx["c"]
        return (
                self.aggregate_by_age(solution, idx_cp) +
                self.aggregate_by_age(solution, idx_c)
        ).max()

    def get_icu_cases(self, solution: np.ndarray) -> float:
        idx_cp = self.c_idx["c"]
        return self.aggregate_by_age(solution, idx_cp).max()

    def get_final_size_dead(self, solution: np.ndarray) -> float:
        idx_d = self.c_idx["d"]
        final_state = solution[-1].reshape((1, -1))
        return self.aggregate_by_age(final_state, idx_d)