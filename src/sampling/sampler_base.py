from abc import ABC, abstractmethod
import os
import json

import numpy as np
from smt.sampling_methods import LHS

import src


class SamplerBase(ABC):
    def __init__(self, sim_obj: src.SimulationNPI, config, target="r0") -> None:
        self.config = config
        self.target = target
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

    def _get_lhs_table(self, model: str, strategy: str, number_of_samples: int = 120000,
                       kappa: float = None, delta: float = 0.5) -> np.ndarray:
        """
        Generate LHS table using selected strategy:
        - 'baseline': upper_bound *= (1 - kappa)
        - 'absolute': [Cij - delta, Cij + delta]
        - 'poisson': [Cij ± 2 * sqrt(Cij / n)]
        :param number_of_samples: Number of LHS samples
        :param kappa: Parameter for 'original' strategy
        :return: delta: Parameter for 'absolute' strategy.
        """
        lower_bound = self.lhs_boundaries["lower"]
        upper_bound = self.lhs_boundaries["upper"]

        if strategy == "baseline":
            upper_bound *= (1 - kappa)

        elif strategy == "absolute":
            lower_bound = np.clip(upper_bound - delta, a_min=0, a_max=None)
            upper_bound = upper_bound + delta

        elif strategy == "poisson":
            if model == "seir":
                n_participants = 67  # POLY-MOD participants for GB (1012 / 15 age groups)
            elif model == "rost":
                # Online 3 week (12208), CATI Survey each month (1500), total 234503
                n_participants = 188  # Using CATI (1500 / 8 age groups)
            else:
                raise ValueError(f"Unknown model '{model}' for poisson strategy.")
            deviation = 2 * np.sqrt(upper_bound / n_participants)
            lower_bound = np.clip(upper_bound - deviation, a_min=0, a_max=None)
            upper_bound = upper_bound + deviation
        else:
            raise ValueError(f"Invalid strategy '{strategy}'. Choose 'baseline', "
                             f"'absolute', or 'poisson'.")

        lhs_table = create_latin_table(n_of_samples=number_of_samples,
                                       lower=lower_bound,
                                       upper=upper_bound)
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
