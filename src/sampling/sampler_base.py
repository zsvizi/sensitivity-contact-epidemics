from abc import ABC, abstractmethod
import os

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
    def run(self, option):
        pass

    @abstractmethod
    def _get_variable_parameters(self):
        pass

    def _get_lhs_table(self, number_of_samples: int = 120000, kappa: float = None) -> np.ndarray:
        # only computes lhs for icu with a_ij
        # Get actual limit matrices
        lower_bound = self.lhs_boundaries["lower"]
        upper_bound = self.lhs_boundaries["upper"]

        if kappa is not None:
            upper_bound *= (1 - kappa)

        # Get LHS tables
        lhs_table = create_latin_table(n_of_samples=number_of_samples,
                                       lower=lower_bound,
                                       upper=upper_bound)
        return lhs_table

    def _save_output(self, output_dict, folder_name):
        # Check if 'output_dict' is empty or not a dictionary
        if output_dict is None or not isinstance(output_dict, dict) or \
                len(output_dict) == 0:
            print("Error: 'output_dict' is empty or not a dictionary. Unable to save.")
            return

        # Create directory for saving calculation outputs
        directory = os.path.join("./sens_data", folder_name)
        os.makedirs(directory, exist_ok=True)

        # Initialize a common filename for all targets
        common_filename = os.path.join(directory,
                                       f"{folder_name}_Hungary_{self.target}_"
                                       f"{'_'.join(self._get_variable_parameters())}.csv")

        # Iterate over the dictionary and save each array separately
        for key, value in output_dict.items():
            if value is None or value.size == 0 or value.ndim not in (1, 2):
                print(f"Error: '{key}' array is empty or has invalid dimensions. "
                      f"Skipping.")
                continue

            # Reshape the array based on the sample size
            sample_size = len(value)
            num_columns = len(value[0])
            reshaped_value = value.reshape((sample_size, num_columns))

            # Save the array with the common filename
            with open(common_filename, "a") as f:
                # Write header if file is empty
                if os.stat(common_filename).st_size == 0:
                    f.write("Target," + ",".join(map(str, range(num_columns))) + "\n")
                # Write target name and array values
                f.write(f"{key}," + ",".join(map(str, reshaped_value.flatten())) + "\n")


def create_latin_table(n_of_samples, lower, upper) -> np.ndarray:
    bounds = np.array([lower, upper]).T
    sampling = LHS(xlimits=bounds)
    return sampling(n_of_samples)
