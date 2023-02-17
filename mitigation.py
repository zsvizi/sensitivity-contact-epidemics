from abc import ABC
import numpy as np
from Target_calculation import TargetCalculator
from prcc import get_contact_matrix_from_upper_triu
from sampler_base import SamplerBase


class Mitigation(SamplerBase, ABC):
    def __init__(self, sim_state: dict, sim_obj, output: TargetCalculator):
        super().__init__(sim_state, sim_obj)

        self.sim_obj = sim_obj
        self.output = output.cm_sim

    def _get_sim_output_cm_entries(self, lhs_sample: np.ndarray):
        # Get output
        cm_sim = get_contact_matrix_from_upper_triu(rvector=lhs_sample,
                                                    age_vector=self.sim_obj.age_vector.reshape(-1,))
        output = self.output()
        output = np.append(lhs_sample, output)
        return list(output)


