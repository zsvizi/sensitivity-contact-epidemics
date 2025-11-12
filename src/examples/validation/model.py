import numpy as np

from src.model.model_base import EpidemicModelBase


class ValidationModel(EpidemicModelBase):
    """
    A validation version of an age-structured SEIRD epidemic model.

    Compartments:
        s   - Susceptible population
        e   - Exposed (infected but not yet infectious)
        i   - Infectious
        r   - Recovered
        d   - Deceased
        inf - Cumulative infections (tracking total cases over time)
    """

    def __init__(self, model_data) -> None:
        """
        Initializes the validation model with model data and define the compartments.

        :param model_data: Model configuration and demographic information
                           (e.g., population, contact matrix, age structure, etc.)
        """
        compartments = ["s", "e", "i", "r", "d", "inf"]
        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        """
        Updates initial compartment values before simulation.

        :param dict iv: Dictionary of initial compartment values for each age group.
        """
        # Introduce one initially exposed individual in the second age group
        iv["e"][1] = 1

        # Total infections at start = infectious + recovered + deceased
        iv.update({
            "inf": iv["i"] + iv["r"] + iv["d"]
        })

        # Adjust susceptible so total across compartments equals total population
        iv.update({
            "s": self.population - (iv["e"] + iv["inf"])
        })

    def get_model(self, xs: np.ndarray, t, ps: dict, cm: np.ndarray) -> np.ndarray:
        """
        Define the system of differential equations governing epidemic dynamics.

        :param np.ndarray xs: Flattened array of compartment values across all age groups.
        :param int t: Current simulation time (unused here, included for ODE solver compatibility).
        :param dict ps: Dictionary of model parameters:
            - beta: Transmission rate
            - alpha: Rate from exposed → infectious
            - gamma: Recovery/death rate (1 / infectious period)
            - p_recovery: Probability of recovery (vs death)
        :param np.ndarray cm: Contact matrix describing average contacts between age groups.

        :return np.ndarray: Array of derivatives (dS/dt, dE/dt, dI/dt, dR/dt, dD/dt, dC/dt)
        """
        s, e, i, r, d, inf = xs.reshape(-1, self.n_age)

        # Compute total force of infection (E does not contribute)
        transmission = ps["beta"] * np.array(i).dot(cm)

        # Differential equations for each compartment
        model_eq_dict = {
            "s": -transmission * s / self.population,  # S'(t)
            "e": s / self.population * transmission - e * ps["alpha"],  # E'(t)
            "i": ps["alpha"] * e - ps["gamma"] * i,  # I'(t)
            "r": ps["p_recovery"] * ps["gamma"] * i,  # R'(t)
            "d": (1 - ps["p_recovery"]) * ps["gamma"] * i,  # D'(t)
            # add compartment to store total infecteds
            "inf": ps["alpha"] * e  # C'(t)
        }

        # Convert from dict to array matching model's internal ordering
        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_infected(self, solution: np.ndarray) -> np.ndarray:
        """
        Compute the total number of currently infected individuals (E + I) over time.

        :param np.ndarray solution: Full time-series solution array from ODE integration.
        :return np.ndarray: Array of total infected counts (summed over all age groups) per time step.
        """
        idx_e = self.c_idx["e"]
        idx_i = self.c_idx["i"]
        return self.aggregate_by_age(solution, idx_e) + \
            self.aggregate_by_age(solution, idx_i)

    def get_epidemic_peak(self, solution: np.ndarray) -> float:
        """
        Determine the peak value of total infected individuals (E + I).

        :param np.ndarray solution: Full time-series solution array from ODE integration.
        :return float: Maximum total infected count across all time points.
        """
        idx_e = self.c_idx["e"]
        idx_i = self.c_idx["i"]
        total_infected = self.aggregate_by_age(solution, idx_e) + \
                         self.aggregate_by_age(solution, idx_i)
        return total_infected.max()

    def get_final_size_dead(self, solution: np.ndarray) -> np.ndarray:
        """
        Compute the total number of deceased individuals at the end of the simulation.

        :param np.ndarray solution: Full time-series solution array from ODE integration.
        :return np.ndarray: Final number of deaths (summed across all age groups).
        """
        state = solution[-1].reshape((1, -1))
        return self.aggregate_by_age(state, self.c_idx["d"])
