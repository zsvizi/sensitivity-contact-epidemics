import numpy as np

import src
from src.prcc.prcc import get_rectangular_matrix_from_upper_triu


class CMCalculatorLockdown:
    """
    Calculates contact matrices (C_ij) during lockdown scenarios using sampled parameters
    and runs corresponding simulations.

    This class reconstructs full contact matrices from sampled upper-triangular
    elements, adjusts them based on lockdown strategies, and executes simulations.
    """

    def __init__(self, sim_obj: src.SimulationNPI) -> None:
        """
        Initializes the lockdown contact matrix calculator.

        :param src.SimulationNPU sim_obj: Simulation object containing configuration,
                                         age structure, and contact matrices.
        """
        self.sim_obj = sim_obj

        # Define lower and upper bounds for Latin Hypercube Sampling (LHS)
        self.lhs_boundaries = {
            "lower": np.zeros(self.sim_obj.upper_tri_size),
            "upper": np.ones(self.sim_obj.upper_tri_size)
        }

    def get_sim_output_cm_entries_lockdown(self, lhs_sample: np.ndarray, calc, strategy: str) -> list:
        """
        Generate a contact matrix based on the provided LHS sample and lockdown strategy,
        then run a simulation and return its results.

        :param np.ndarray lhs_sample: Vector of sampled upper-triangular values from LHS
        :param calc: Simulation calculation object (providing a `get_output(cm)` method)
        :param str strategy: Sampling strategy used (e.g. "poisson")

        :return list: Combined list of simulation outputs and the flattened upper-triangular
                  values of the sampled total contact matrix
        """
        if strategy == "baseline":
            # Construct the upper-triangular ratio matrix from the LHS vector
            ratio_matrix = get_rectangular_matrix_from_upper_triu(
                rvector=lhs_sample,
                matrix_size=self.sim_obj.n_ag
            )

            # Reduce "non-home" contacts according to ratio_matrix,
            # keep "home" contacts unchanged
            cm_sim = (1 - ratio_matrix) * (self.sim_obj.contact_matrix - self.sim_obj.contact_home)
            cm_sim += self.sim_obj.contact_home
        else:
            # Construct the full contact matrix (upper-triangular filled symmetrically)
            other_cm_sim = get_rectangular_matrix_from_upper_triu(
                rvector=lhs_sample,
                matrix_size=self.sim_obj.n_ag
            )

            # `other_cm_sim` currently represents the sampled (with LHS) total contacts per age group
            # To convert these to average per-person contact rates, divide by population size
            other_cm_sim /= self.sim_obj.age_vector

            # Add back home contacts to get the total contact matrix
            cm_sim = other_cm_sim + self.sim_obj.contact_home

        # Run the simulation with the generated contact matrix
        output = calc.get_output(cm=cm_sim)

        # Compute total contacts (scaled by population) and extract upper-triangular elements
        cm_total_sim = (cm_sim * self.sim_obj.age_vector)[self.sim_obj.upper_tri_indexes]

        # Append the simulation output to the upper-triangular contact values
        output = np.append(cm_total_sim, output)

        return list(output)
