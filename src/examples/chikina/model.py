import numpy as np

from src.model.model_base import EpidemicModelBase


class SirModel(EpidemicModelBase):
    """
    Extended age-structured SIR-type epidemic model.

    This model contains the following compartments for each age group:
    - s: Susceptible
    - i: Infectious
    - cp: Clinical presymptomatic
    - c: Clinical symptomatic
    - r: Recovered
    - d: Dead
    - inf: Cumulative infections
    - hosp: Cumulative hospitalisations (proxy)
    - icu: Cumulative ICU admissions

    The model implements the system of ODEs through `get_model()`, and
    provides helper methods to compute epidemiological metrics.

    :param dict model_data: Dictionary containing parameters, initial values,
                            population, contact matrix, etc.
    """

    def __init__(self, model_data) -> None:
        compartments = ["s", "i", "cp", "c", "r", "d", "inf", "hosp", "icu"]
        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        """
        Updates the initial values' dictionary.

        This function:
        - Forces 1 initial infectious case in age group 3.
        - Copies values for compartments that should remain unchanged.
        - Recomputes susceptible values to ensure population consistency.

        :param dict iv: Dictionary of initial compartment values per age group.
        """
        # Seed one initial infection in a specific age group
        iv["i"][3] = 1

        # Pass-through compartments that should not be altered
        keys_to_copy = ["c", "r", "d", "inf", "hosp", "icu"]
        iv.update({key: iv[key] for key in keys_to_copy})

        # Recompute susceptible population:
        # S = N − (I + Cp + C + R + D)
        iv.update({
            "s": self.population - sum(
                iv["i"] + iv["cp"] + iv["c"] + iv["r"] + iv["d"]
            )
        })

    def get_model(self, xs: np.ndarray, t: int, ps: dict, cm: np.ndarray) -> np.ndarray:
        """
        Computes the time derivatives for the ODE system.

        :param np.ndarray xs: Flattened state vector for all age groups.
                              Shape: (n_age * n_compartments,)
        :param int t: Current time
        :param dict ps: Parameter dictionary (alpha_i, xi, beta, susc, etc.)
        :param np.ndarray cm: Contact matrix (per capita contact rates).

        :return np.ndarray: Derivatives in the same flattened structure.
        """
        beta = ps["beta"]

        # Reshape xs to separate compartments by age: (n_compartments, n_age)
        s, i, cp, c, r, d, inf, hosp, icu = xs.reshape(-1, self.n_age)

        # Force of infection: per-age-group transmission probability
        # (i + cp) produce infections weighted by contact matrix
        transmission = beta * np.dot((i + cp), cm)

        # ODE system for each compartment
        sir_eq_dict = {
            "s": -ps["susc"] * transmission * s / self.population,

            "i": (
                ps["susc"] * (1 - ps["xi"]) * (transmission * s / self.population)
                - ps["alpha_i"] * i
            ),

            "cp": (
                ps["susc"] * ps["xi"] * (transmission * s / self.population)
                - ps["alpha_p"] * cp
            ),

            "c": ps["alpha_p"] * cp - ps["alpha_c"] * c,

            "r": ps["alpha_i"] * i + ps["alpha_c"] * c,

            "d": ps["gamma_d"] * r,  # Deaths from recovered (transition-from-R model)

            # Cumulative infections
            "inf": (
                ps["susc"] * (1 - ps["xi"]) * (transmission * s / self.population)
                + ps["susc"] * ps["xi"] * (transmission * s / self.population)
            ),

            # Cumulative hospitalisation proxy (duplicated infection term?)
            "hosp": (
                ps["susc"] * ps["xi"] * (transmission * s / self.population)
                + ps["susc"] * ps["xi"] * (transmission * s / self.population)
            ),

            # Cumulative ICU transitions (presymptomatic -> clinical)
            "icu": ps["alpha_p"] * cp,
        }

        # Convert dict of compartment arrays into a single flattened vector
        return self.get_array_from_dict(comp_dict=sir_eq_dict)

    def get_infected(self, solution: np.ndarray) -> np.ndarray:
        """
        Gets the total number of currently infected individuals over time.

        Infected = I + C_p + C

        :param np.ndarray solution: ODE solution array (timesteps × state_vector).
        :return np.ndarray: Time series of total infected individuals.
        """
        idx_i = self.c_idx["i"]
        idx_cp = self.c_idx["cp"]
        idx_c = self.c_idx["c"]

        return (
            self.aggregate_by_age(solution, idx_i)
            + self.aggregate_by_age(solution, idx_cp)
            + self.aggregate_by_age(solution, idx_c)
        )

    def get_epidemic_peak(self, solution: np.ndarray) -> float:
        """
        Return the peak number of infected individuals.

        :param np.ndarray solution: ODE solution array.
        :return float: Maximum value of I + Cp + C.
        """
        idx_i = self.c_idx["i"]
        idx_cp = self.c_idx["cp"]
        idx_c = self.c_idx["c"]

        return (
            self.aggregate_by_age(solution, idx_i)
            + self.aggregate_by_age(solution, idx_cp)
            + self.aggregate_by_age(solution, idx_c)
        ).max()

    def get_hospital_peak(self, solution: np.ndarray) -> float:
        """
        Compute the peak number of individuals in the "clinical" stages
        (proxy for hospital load).

        :param np.ndarray solution: ODE solution array.
        :return float: Maximum of Cp + C.
        """
        idx_cp = self.c_idx["cp"]
        idx_c = self.c_idx["c"]

        return (
            self.aggregate_by_age(solution, idx_cp)
            + self.aggregate_by_age(solution, idx_c)
        ).max()

    def get_icu_cases(self, solution: np.ndarray) -> float:
        """
        Return the maximum number of ICU cases.

        :param np.ndarray solution: ODE solution array.
        :return float: Peak ICU occupancy.
        """
        idx_cp = self.c_idx["c"]
        return self.aggregate_by_age(solution, idx_cp).max()

    def get_final_size_dead(self, solution: np.ndarray) -> np.ndarray:
        """
        Computes the final epidemic size for the mortality compartment.

        :param np.ndarray solution: ODE solution (timesteps * state_vector).
        :return float: Total number of deaths at the final time.
        """
        idx_d = self.c_idx["d"]
        final_state = solution[-1].reshape((1, -1))
        return self.aggregate_by_age(final_state, idx_d)
