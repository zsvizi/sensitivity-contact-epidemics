import numpy as np

from src.simulation_base import SimulationBase
from src.target_calculation import TargetCalculator
from src.prcc import get_rectangular_matrix_from_upper_triu


class CMCalculatorLockdown:
    def __init__(self, sim_state, data_tr: SimulationBase) -> None:
        self.data_tr = data_tr
        self.sim_state = sim_state

        self.lhs_boundaries = \
            {  # Contact matrix entry level approach, full scale approach (old name: "home")
                "lockdown": {"lower": np.zeros(self.data_tr.upper_tri_size),
                             "upper": np.ones(self.data_tr.upper_tri_size)}
            }

    def get_sim_output_cm_entries_lockdown(self, lhs_sample: np.ndarray):
        # Get ratio matrix
        ratio_matrix = get_rectangular_matrix_from_upper_triu(rvector=lhs_sample,
                                                              matrix_size=self.data_tr.n_ag)
        # Get modified full contact matrix
        cm_sim = (1 - ratio_matrix) * (self.data_tr.contact_matrix - self.data_tr.contact_home)   # first condition
        cm_sim += self.data_tr.contact_home
        # Get output from target calculator
        tar = TargetCalculator(data_tr=self.data_tr, sim_state=self.sim_state)
        output = tar.get_output(cm_sim=cm_sim)
        cm_total_sim = (cm_sim * self.data_tr.age_vector)[self.data_tr.upper_tri_indexes]
        output = np.append(cm_total_sim, output)
        return list(output)
