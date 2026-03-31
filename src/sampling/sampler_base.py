from abc import ABC, abstractmethod
import os
import json

import numpy as np
from scipy.stats import norm
from smt.sampling_methods import LHS

import src


class SamplerBase(ABC):
    """
    Abstract base class for different sampling strategies used in sensitivity analyses.
    """

    def __init__(self, sim_obj: src.SimulationNPI) -> None:
        """
        Initialize the base sampler with simulation parameters and state variables.

        :param src.SimulationNPI sim_obj: Simulation object containing configuration, state variables, and matrices.
        """
        self.config = sim_obj.config
        self.sim_obj = sim_obj
        self.base_r0 = sim_obj.sim_state["base_r0"]
        self.beta = sim_obj.sim_state["beta"]

        self.r0generator = sim_obj.sim_state["r0generator"]
        self.lhs_boundaries = None

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _get_variable_parameters(self):
        pass

    def _get_lhs_table(self, strategy: str, number_of_samples: int = 120000,
                       kappa: float = None, delta: float = 1e2,
                       n: int = 188, contact_dispersion: float = 0.5) -> np.ndarray:
        """
        Generate a Latin Hypercube Sampling (LHS) table for contact matrices based on a chosen strategy.

        Supported strategies:
            - 'baseline': samples contact reduction ratios in [0, 1 - kappa].
            - 'absolute': samples contact values as C_ij ± delta
            - 'relative': samples contact values as C_ij ± 20%
            - 'poisson': samples contact values from N(C_ij, sqrt(C_ij / n))
            - 'nbinom': samples contact values from N(C_ij, sqrt((C_ij + alpha * C_ij^2) / n)) where alpha is
            the `contact_dispersion` parameter.

        :param str strategy: Sampling strategy to use
        :param int number_of_samples: Number of samples to generate
        :param float kappa: Reduction ratio for baseline sampling
        :param float delta: Absolute variation range for 'absolute' strategy
        :param float contact_dispersion: The dispersion parameter used when sampling contact matrix values
        :param int n: The assumed survey sample size (number of participants) used to calculate the variance
                      during contact matrix sampling.

        :return np.ndarray: The generated LHS sample table
        """
        lower_bound_base = self.lhs_boundaries["lower"]
        upper_bound_base = self.lhs_boundaries["upper"]
        n_params = self.sim_obj.upper_tri_size

        if strategy == "baseline":
            if kappa is None:
                raise ValueError("Kappa must be provided for 'baseline' strategy.")
            lower_bound = lower_bound_base
            upper_bound = upper_bound_base * (1 - kappa)
            return create_latin_table(
                n_of_samples=number_of_samples,
                lower=lower_bound,
                upper=upper_bound
            )

        # Compute the non-home ('other') contact matrix
        contact_other_mtx = self.sim_obj.contact_matrix - self.sim_obj.contact_home
        # To maintain symmetry (c_ij * N_i = c_ji * N_j) after sampling,
        # we scale the contact matrix by the population vector.
        # The scaling will be reversed later, after sampling - see cm_calculator_lockdown.py
        contact_other_mtx_total = contact_other_mtx * self.sim_obj.age_vector

        # Extract upper-triangular values only
        contact_other_values = contact_other_mtx_total[self.sim_obj.upper_tri_indexes]

        # Generate base LHS samples from [0, 1]
        lhs_table = create_latin_table(
            n_of_samples=number_of_samples,
            lower=[0.0] * n_params,
            upper=[1.0] * n_params
        )

        if strategy == "absolute":
            # Sample around delta absolute difference
            lower_bound = np.clip(contact_other_values - delta, 0, None)
            upper_bound = contact_other_values + delta
            for i in range(n_params):
                lhs_table[:, i] = lower_bound[i] + (upper_bound[i] - lower_bound[i]) * lhs_table[:, i]

        elif strategy == "relative":
            # Sample 20% relative variation
            lower_bound = contact_other_values * 0.8
            upper_bound = contact_other_values * 1.2
            for i in range(n_params):
                lhs_table[:, i] = lower_bound[i] + (upper_bound[i] - lower_bound[i]) * lhs_table[:, i]

        elif strategy in ["poisson", "nbinom"]:
            contact_values = np.clip(contact_other_values, a_min=1e-6, a_max=None)
            # In case of "poisson" strategy the variance will be mu/n
            dispersion = contact_dispersion if strategy == "nbinom" else 0.0

            for i in range(n_params):
                mu = contact_values[i]
                variance = (mu + dispersion * (mu ** 2)) / n
                std_dev = np.sqrt(variance)

                lhs_table[:, i] = norm(
                    loc=mu, scale=std_dev
                ).ppf(lhs_table[:, i])

                # Ensure all sampled contact values are non-negative
                lhs_table[:, i] = np.maximum(lhs_table[:, i], 0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return lhs_table

    def _save_output(self, output, folder_name: str):
        """
        Saves a NumPy array of sampled data to a CSV file.

        :param np.ndarray output: The data to be saved.
        :param str folder_name: Name of the subdirectory where results are stored.
        """
        directory = os.path.join("./sens_data", folder_name)
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(
            directory,
            f"{folder_name}_Hungary_" + "_".join(self._get_variable_parameters())
        )

        np.savetxt(fname=filename + ".csv", X=output, delimiter=";")

    def _save_output_json(self, folder_name):
        """
        Saves the configuration parameter mapping (used variable indices) as a JSON file.

        :param str folder_name: Name of the subdirectory where results are stored.
        """
        directory = os.path.join("./sens_data", folder_name)
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(
            directory,
            f"{folder_name}_Hungary_" + "_".join(self._get_variable_parameters())
        )

        value = {}
        i = 0
        for k, v in self.sim_obj.config.items():
            if v:
                value[k] = self.sim_obj.upper_tri_size + i
                i += 1

        with open(filename + ".json", "w") as json_file:
            json.dump(value, json_file)


def create_latin_table(n_of_samples: int, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Generate a Latin Hypercube Sampling (LHS) table within specified bounds.

    :param int n_of_samples: Number of samples to generate.
    :param np.ndarray lower: Lower bounds for each parameter.
    :param np.ndarray upper: Upper bounds for each parameter.

    :return np.ndarray: A 2D array of shape (n_of_samples, n_parameters) containing LHS samples.
    """
    bounds = np.array([lower, upper]).T
    sampling = LHS(xlimits=bounds, random_state=42)

    return sampling(n_of_samples)
