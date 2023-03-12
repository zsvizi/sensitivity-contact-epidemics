import numpy as np
from src.target_calculation import TargetCalculator
from src.prcc import get_rectangular_matrix_from_upper_triu

from src.data_transformer import Transformer


class Lockdown:
    def __init__(self, sim_obj: Transformer, lhs_sample: np.ndarray) -> None:
        self.sim_obj = sim_obj
        self.lhs_sample = lhs_sample
        self.output = []

        self._get_sim_output_cm_entries_lockdown(lhs_sample=lhs_sample)

        self.lhs_boundaries = \
            {  # Contact matrix entry level approach, full scale approach (old name: "home")
                "lockdown": {"lower": np.zeros(self.sim_obj.upper_tri_size),
                             "upper": np.ones(self.sim_obj.upper_tri_size)}
            }

    def _get_sim_output_cm_entries_lockdown(self, lhs_sample: np.ndarray):
        # Get ratio matrix
        ratio_matrix = get_rectangular_matrix_from_upper_triu(rvector=lhs_sample,
                                                              matrix_size=self.sim_obj.n_ag)
        # Get modified full contact matrix
        cm_sim = (1 - ratio_matrix) * (self.sim_obj.contact_matrix - self.sim_obj.contact_home)   # first condition
        cm_sim += self.sim_obj.contact_home
        # Get output from target calculator
        tar = TargetCalculator(sim_obj=self.sim_obj, cm_sim=cm_sim)
        output = tar.output
        cm_total_sim = (cm_sim * self.sim_obj.age_vector)[self.sim_obj.upper_tri_indexes]
        output = np.append(cm_total_sim, output)
        self.output = output
        return list(output)
