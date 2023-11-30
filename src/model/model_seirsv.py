import numpy as np
import math

from src.model.model_base import EpidemicModelBase


class SeirSVModel(EpidemicModelBase):
    def __init__(self, model_data, uk_ps, uk_cm) -> None:
        self.uk_ps = uk_ps
        self.uk_cm = uk_cm
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
        z = 1 + self.uk_ps["q"] * np.sin(2 * math.pi * (t - self.uk_ps["t_offset"]) / 365)

        # calculate the age dependent force of infection
        lambda_i = z * np.array(i).dot(self.uk_cm)

        seirSV_eq_dict = {
            "s": self.uk_ps["omega_v"] * v + self.uk_ps["omega_i"] * r - s / self.uk_population * [self.uk_ps["mu_i"] +
                                                                                                   self.uk_ps["psi_i"] +
                                                                                                   lambda_i],  # S'(t)
            "e": lambda_i * s / self.uk_population - e * [self.uk_ps["mu_i"] + self.uk_ps["gamma"]],  # E'(t)
            "i": self.uk_ps["gamma"] * e - i * [self.uk_ps["mu_i"] + self.uk_ps["rho"]],  # I'(t)
            "r": self.uk_ps["rho"] * i - r * [self.uk_ps["mu_i"] - self.uk_ps["omega_i"]],  # R'(t)
            "v": self.uk_ps["psi_i"] * s / self.uk_population - v * [self.uk_ps["mu_i"] +
                                                                     self.uk_ps["omega_v"]],  # V'(t)
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
