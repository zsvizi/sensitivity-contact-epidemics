from abc import ABC, abstractmethod
import os

import numpy as np
from smt.sampling_methods import LHS

import src


class SamplerBase(ABC):
    def __init__(self, sim_obj: src.SimulationNPI) -> None:
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
        print("Simulation for", number_of_samples,
              "samples (", "-".join(self._get_variable_parameters()), ")")
        return lhs_table

    def _save_output(self, output, folder_name):
        # Create directories for saving calculation outputs
        os.makedirs("./sens_data", exist_ok=True)

        # Save LHS output
        os.makedirs("./sens_data/" + folder_name, exist_ok=True)
        filename = "./sens_data/" + folder_name + "/" + folder_name + "_Hungary_" + \
                   "_".join(self._get_variable_parameters())
        np.savetxt(fname=filename + ".csv", X=np.asarray(output), delimiter=";")


def create_latin_table(n_of_samples, lower, upper) -> np.ndarray:
    bounds = np.array([lower, upper]).T
    sampling = LHS(xlimits=bounds)
    return sampling(n_of_samples)
