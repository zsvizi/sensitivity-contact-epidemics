import numpy as np
import math

from src.model.model_base import EpidemicModelBase


class SeirSVModel(EpidemicModelBase):
    def __init__(self, model_data, country: str = "UK") -> None:
        self.country = country
        compartments = ["s", "e", "i", "r", "v"]

        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        iv["i"][3] = 1
        iv.update({"e": iv["e"], "r": iv["r"], "v": iv["v"]})

        iv.update({"s": self.population - (iv["e"] + iv["i"] + iv["r"] + iv["v"])})

    def get_model(self, xs: np.ndarray, t: int, ps, cm: np.ndarray) -> np.ndarray:
        """
        Implemented the transmission model in Appendix A, section 2 i.e. A.2
        The model uses parameters corresponding to Influenza A.
        :param xs: values of all the compartments as a numpy array
        :param t: number of days since the start of the simulation.
        :param ps: uk parameters stored as a dictionary
        :param cm: uk contact matrix
        :return: seirSV model as an array
        """
        s, e, i, r, v = xs.reshape(-1, self.n_age)

        # calculate a sine wave function emulating the seasonal  fluctuation in the force of infection:
        z = 1 + ps["q"] * np.sin(2 * math.pi * (t - ps["t_offset"]) / 365)

        # calculate the age dependent force of infection
        lambda_i = z * np.array(i).dot(cm)

        seirSV_eq_dict = {
            "s": ps["omega_v"] * v + ps["omega_i"] * r - s / self.population * [ps["mu_i"] +
                                                                                ps["psi_i"] + lambda_i],  # S'(t)
            "e": lambda_i * s / self.population - e * [ps["mu_i"] + ps["gamma"]],  # E'(t)
            "i": ps["gamma"] * e - i * [ps["mu_i"] + ps["rho"]],  # I'(t)
            "r": ps["rho"] * i - r * [ps["mu_i"] - ps["omega_i"]],  # R'(t)
            "v": ps["psi_i"] * s / self.population - v * [ps["mu_i"] + ps["omega_v"]],  # V'(t)
        }

        return self.get_array_from_dict(comp_dict=seirSV_eq_dict)

    def get_infected(self, solution) -> np.ndarray:
        idx = self.c_idx["i"]
        return self.aggregate_by_age(solution, idx)

    def get_recovered(self, solution) -> np.ndarray:
        idx = self.c_idx["r"]
        return self.aggregate_by_age(solution, idx)

    def get_vaccinated(self, solution) -> np.ndarray:
        idx = self.c_idx["v"]
        return self.aggregate_by_age(solution, idx)
