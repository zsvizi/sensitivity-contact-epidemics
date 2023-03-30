import numpy as np

import src
from src.prcc import get_rectangular_matrix_from_upper_triu
from src.sampling.target_calculator import TargetCalculator


class CMCalculatorLockdownTypewise:
    def __init__(self, sim_state, sim_obj: src.SimulationNPI) -> None:
        self.sim_obj = sim_obj
        self.sim_state = sim_state

        self.lhs_boundaries = \
            {
                # Contact matrix entry level approach, full scale approach (old name: "home")
                "lockdown_3": {"lower": np.zeros(3 * self.sim_obj.upper_tri_size),
                               "upper": np.ones(3 * self.sim_obj.upper_tri_size)}
            }

    def get_sim_output_cm_entries_lockdown_3(self, lhs_sample: np.ndarray):
        # Get number of elements in the upper triangular matrix
        u_t_s = self.sim_obj.upper_tri_size
        # Get ratio matrices
        ratio_matrix_school = get_rectangular_matrix_from_upper_triu(rvector=lhs_sample[:u_t_s],  # first 136
                                                                     matrix_size=self.sim_obj.n_ag)
        ratio_matrix_work = get_rectangular_matrix_from_upper_triu(rvector=lhs_sample[u_t_s:2 * u_t_s],  # from 136-272
                                                                   matrix_size=self.sim_obj.n_ag)
        ratio_matrix_other = get_rectangular_matrix_from_upper_triu(rvector=lhs_sample[2 * u_t_s:],  # from 272
                                                                    matrix_size=self.sim_obj.n_ag)
        # Get modified contact matrices per layers
        cm_sim_school = (1 - ratio_matrix_school) * self.sim_obj.data.contact_data["school"]
        cm_sim_work = (1 - ratio_matrix_work) * self.sim_obj.data.contact_data["work"]
        cm_sim_other = (1 - ratio_matrix_other) * self.sim_obj.data.contact_data["other"]
        # Get full contact matrix
        cm_sim = self.sim_obj.contact_home + cm_sim_school + cm_sim_work + cm_sim_other
        # Get output from target calculator
        targ = TargetCalculator(sim_obj=self.sim_obj, sim_state=self.sim_state)
        output = targ.get_output(cm_sim=cm_sim)
        cm_total_sim = (cm_sim * self.sim_obj.age_vector)[self.sim_obj.upper_tri_indexes]
        output = np.append(cm_total_sim, output)
        return list(output)
