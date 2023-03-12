from abc import ABC, abstractmethod
import os

import numpy as np
from smt.sampling_methods import LHS
from src.data_transformer import Transformer


class SamplerBase(ABC):
    def __init__(self, sim_state: dict, sim_obj: Transformer) -> None:
        self.sim_obj = sim_obj
        self.base_r0 = sim_state["base_r0"]
        self.beta = sim_state["beta"]
        self.type = sim_state["type"]

        self.r0generator = sim_state["r0generator"]
        self.lhs_boundaries = None
        self._get_lhs_table()

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _get_variable_parameters(self):
        pass

    def _get_lhs_table(self, number_of_samples: int = 40000, kappa=None, sim_obj=None) -> np.ndarray:
        # only computes lhs for icu with a_ij
        # Get actual limit matrices
        lower_bound = self.lhs_boundaries[self.type]["lower"]
        upper_bound = self.lhs_boundaries[self.type]["upper"]

        if kappa is not None:
            upper_bound *= (1-kappa)

        if sim_obj is not None:
            p_icr = (1 - sim_obj.params['p']) * sim_obj.params['h'] * sim_obj.params['xi']
            a = np.zeros((16, 16))
            for i in range(16):
                for j in range(16):
                    a[i, j] = p_icr[i] * p_icr[j] / np.sum(p_icr) ** 2

        # Get LHS tables
        lhs_table = create_latin_table(n_of_samples=number_of_samples,
                                       lower=lower_bound,
                                       upper=upper_bound)
        print("Simulation for", number_of_samples,
              "samples (", "-".join(self._get_variable_parameters()), ")")
        self.lhs_sample = lhs_table
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