import math
import numpy as np
from src.model.model_base import EpidemicModelBase


class SeirUK(EpidemicModelBase):
    """
    SEIR model with seasonality used for the United Kingdom epidemic simulations.

    Compartments:
        s : Susceptible individuals.
        e : Exposed (infected but not yet infectious).
        i : Infectious individuals.
        r : Recovered individuals.
        c : Cumulative number of infections.
    """

    def __init__(self, model_data) -> None:
        """
        Initializes the SEIR model with UK-specific configuration.

        :param model_data: Model configuration and demographic information
                           (e.g., population, contact matrix, age structure, etc.)
        """
        compartments = ["s", "e", "i", "r", "c"]
        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict) -> None:
        """
        Sets the initial values for model compartments.

        :param dict iv: Dictionary containing initial compartment values for all age groups.
        """
        iv["i"][3] = 1  # Seed infection in the 4th age group
        iv.update({"e": iv["e"], "r": iv["r"]})

        # Ensure population conservation: S = N - (E + I + R + C)
        iv.update({"s": self.population - (iv["e"] + iv["i"] + iv["r"] + iv["c"])})

    def get_model(self, xs: np.ndarray, t: int, ps: dict, cm: np.ndarray) -> np.ndarray:
        """
        Computes the differential equations for the SEIR model.
        :param np.ndarray xs: Current state vector containing compartment values for all age groups.
        :param int t: Current time (in days)
        :param dict ps: Model parameters, including:
            - beta: transmission rate
            - gamma: progression rate from exposed to infectious
            - rho: recovery rate
            - q: amplitude of seasonal variation
            - t_offset: phase shift for seasonal forcing
        :param np.ndarray cm: Contact matrix between age groups.

        :return np.ndarray: The derivatives of all compartments as a flattened array.
        """
        # Unpack compartments according to order in self.compartments
        s, e, i, r, c = xs.reshape(-1, self.n_age)

        # Force of infection: transmission proportional to contacts with infectious individuals
        transmission = ps["beta"] * np.array(i).dot(cm)

        # Apply seasonal forcing: transmission varies sinusoidally over the year
        z = 1 + ps["q"] * np.sin(2 * math.pi * (t - ps["t_offset"]) / 365)
        transmission *= z

        # Define the SEIR system of ODEs
        model_eq_dict = {
            "s": -transmission * s / self.population,  # S'(t)
            "e": s / self.population * transmission - e * ps["gamma"],  # E'(t)
            "i": ps["gamma"] * e - ps["rho"] * i,  # I'(t)
            "r": ps["rho"] * i,  # R'(t)
            "c": s / self.population * transmission + ps["gamma"] * e  # C'(t)
        }

        # Return state derivatives as a flattened array
        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_infected(self, solution: np.ndarray) -> np.ndarray:
        """
        Computes the total number of infectious individuals over time.

        :param np.ndarray solution: Solution matrix of model states over time
        :return np.ndarray: Time series of total infectious individuals (summed over age groups)
        """
        idx = self.c_idx["i"]
        return self.aggregate_by_age(solution, idx)

    def get_recovered(self, solution: np.ndarray) -> np.ndarray:
        """
        Computes the total number of recovered individuals over time.

        :param np.ndarray solution: Solution matrix of model states over time
        :return np.ndarray: Time series of total recovered individuals (summed over age groups)
        """
        idx = self.c_idx["r"]
        return self.aggregate_by_age(solution, idx)
