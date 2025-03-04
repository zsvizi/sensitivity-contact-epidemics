import numpy as np

import src
from src.prcc.prcc import get_rectangular_matrix_from_upper_triu


class CMCalculatorLockdown:
    def __init__(self, sim_obj: src.SimulationNPI) -> None:
        self.sim_obj = sim_obj

        self.lhs_boundaries = \
            {"lower": np.zeros(self.sim_obj.upper_tri_size),
             "upper": np.ones(self.sim_obj.upper_tri_size)}

    def get_sim_output_cm_entries_lockdown(self, lhs_sample: np.ndarray, calc):
        # Get ratio matrix
        ratio_matrix = get_rectangular_matrix_from_upper_triu(
            rvector=lhs_sample,
            matrix_size=self.sim_obj.n_ag
        )
        # Get modified full contact matrix
        # first condition
        cm_sim = (1 - ratio_matrix) * (self.sim_obj.contact_matrix - self.sim_obj.contact_home)
        cm_sim += self.sim_obj.contact_home
        # Get output from target calculator
        # tar = R0TargetCalculator(sim_obj=self.sim_obj, sim_state=self.sim_state)
        output = calc.get_output(cm=cm_sim)
        cm_total_sim = (cm_sim * self.sim_obj.age_vector)[self.sim_obj.upper_tri_indexes]
        output = np.append(cm_total_sim, output)  # 136
        return list(output)
