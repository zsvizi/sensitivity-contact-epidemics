from abc import ABC
import numpy as np
from Target_calculation import TargetCalculator
from prcc import get_rectangular_matrix_from_upper_triu
from sampler_base import SamplerBase


class Lockdown(SamplerBase, ABC):
    def __init__(self, sim_state: dict, sim_obj, output: TargetCalculator):
        super().__init__(sim_state, sim_obj)

        self.sim_obj = sim_obj
        self.output = output.cm_sim

        self.upper_tri_size = int((self.sim_obj.no_ag + 1) * self.sim_obj.no_ag / 2)  # 136 elements n(n+1)/2 = 17 * 8

    def _get_sim_output_cm_entries_lockdown(self, lhs_sample: np.ndarray):
        # Get ratio matrix
        ratio_matrix = get_rectangular_matrix_from_upper_triu(rvector=lhs_sample,
                                                              matrix_size=self.sim_obj.no_ag)
        # Get modified full contact matrix
        cm_sim = (1 - ratio_matrix) * (self.sim_obj.contact_matrix - self.sim_obj.contact_home)   # first condition
        cm_sim += self.sim_obj.contact_home
        # Get output
        output = self.output
        cm_total_sim = (cm_sim * self.sim_obj.age_vector)[self.sim_obj.upper_tri_indexes]
        output = np.append(cm_total_sim, output)
        return list(output)


