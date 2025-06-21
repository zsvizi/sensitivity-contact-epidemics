from abc import ABC, abstractmethod
import os
import json

import numpy as np
from scipy.stats import norm
from smt.sampling_methods import LHS

import src


class SamplerBase(ABC):
    def __init__(self, sim_obj: src.SimulationNPI) -> None:
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

    @staticmethod
    def apply_inverse_cdf(u, dist_type="tent"):
        if dist_type == "tent":
            return np.where(u < 0.5, np.sqrt(u / 2), 1 - np.sqrt(1 - u))
        elif dist_type == "narrow_tent":
            return np.where(u < 0.5,
                            (1 + np.sqrt(2 * u)) / 4,
                            (3 - np.sqrt(2 * (1 - u))) / 4)
        else:
            return u

    def _get_lhs_table(self, model: str, strategy: str, number_of_samples: int = 120000,
                       kappa: float = None, delta: float = 0.5,
                       dist_type: str = "tent") -> np.ndarray:
        """
        Generate LHS table using selected strategy:
        - 'baseline': sample reduction ratios in [0, 1 - kappa]
        - 'absolute': sample Cij ± delta
        - 'relative': sample Cij ± 50%
        - 'poisson': sample from Normal(Cij, sqrt(Cij / n))
        """
        lower_bound_base = self.lhs_boundaries["lower"]
        upper_bound_base = self.lhs_boundaries["upper"]
        n_params = self.sim_obj.upper_tri_size

        if strategy == "baseline":
            lhs_table = create_latin_table(n_of_samples=number_of_samples,
                                           lower=[0.0] * n_params,
                                           upper=[1.0] * n_params)
            if kappa is None:
                lhs_table = self.apply_inverse_cdf(u=lhs_table, dist_type=dist_type)
            else:
                lower_bound = lower_bound_base
                upper_bound = upper_bound_base * (1 - kappa)
                lhs_table = create_latin_table(n_of_samples=number_of_samples,
                                               lower=lower_bound,
                                               upper=upper_bound)
            return lhs_table

        # Compute "other" contact matrix (non-home), upper triangle only
        contact_other = self.sim_obj.contact_matrix - self.sim_obj.contact_home
        contact_other_values = contact_other[self.sim_obj.upper_tri_indexes]

        # Sample from [0, 1], then transform
        lhs_table = create_latin_table(n_of_samples=number_of_samples,
                                       lower=[0.0] * n_params,
                                       upper=[1.0] * n_params)

        if strategy == "absolute":
            lower_bound = np.clip(contact_other_values - delta, 0, None)
            upper_bound = contact_other_values
            for i in range(n_params):
                lhs_table[:, i] = lower_bound[i] + (upper_bound[i] - lower_bound[i]) * \
                                  lhs_table[:, i]
        elif strategy == "relative":
            lower_bound = contact_other_values * 0.5
            upper_bound = contact_other_values
            for i in range(n_params):
                lhs_table[:, i] = lower_bound[i] + (upper_bound[i] - lower_bound[i]) * \
                                  lhs_table[:, i]
        elif strategy == "poisson":
            if model == "seir":
                n_participants = 67
            elif model in ["rost_maszk", "rost_prem", "validation"]:
                n_participants = 188
            else:
                raise ValueError(f"Unknown model '{model}' for poisson strategy.")

            # Avoid division by zero and NaNs by clipping to a small epsilon
            contact_values = np.clip(contact_other_values, 1e-6, None)
            std = np.sqrt(contact_values / n_participants)
            for i in range(n_params):
                lhs_table[:, i] = norm(loc=contact_other_values[i],
                                       scale=std[i]
                                  ).ppf(lhs_table[:, i])
                lhs_table[:, i] = np.maximum(lhs_table[:, i], 0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return lhs_table

    def _save_output(self, output, folder_name):
        # Create directories for saving calculation outputs
        directory = os.path.join("./sens_data", folder_name)
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"{folder_name}_Hungary_" +
                                "_".join(self._get_variable_parameters()))

        # Save NumPy array as CSV file
        np.savetxt(fname=filename + ".csv", X=output, delimiter=";")

    def _save_output_json(self, folder_name):
        directory = os.path.join("./sens_data", folder_name)
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"{folder_name}_Hungary_" +
                                "_".join(self._get_variable_parameters()))
        value = dict()
        i = 0
        for k, v in self.sim_obj.config.items():
            if v:
                value[k] = self.sim_obj.upper_tri_size + i
                i += 1

        with open(filename + ".json", "w") as json_file:
            json.dump(value, json_file)


def create_latin_table(n_of_samples, lower, upper) -> np.ndarray:
    bounds = np.array([lower, upper]).T
    sampling = LHS(xlimits=bounds)
    return sampling(n_of_samples)
