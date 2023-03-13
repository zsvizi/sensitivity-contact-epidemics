import numpy as np

from src.data_transformer import DataTransformer
from src.prcc import get_rectangular_matrix_from_upper_triu
from src.target_calculation import TargetCalculator


class CMCalculatorLockdownTypewise:
    def __init__(self, data_tr: DataTransformer) -> None:
        self.data_tr = data_tr
        self.output = []

        self.lhs_boundaries = \
            {
                # Contact matrix entry level approach, full scale approach (old name: "home")
                "lockdown_3": {"lower": np.zeros(3 * self.data_tr.upper_tri_size),
                               "upper": np.ones(3 * self.data_tr.upper_tri_size)}
            }

    def get_sim_output_cm_entries_lockdown_3(self, lhs_sample: np.ndarray):
        # Get number of elements in the upper triangular matrix
        u_t_s = self.data_tr.upper_tri_size
        # Get ratio matrices
        ratio_matrix_school = get_rectangular_matrix_from_upper_triu(rvector=lhs_sample[:u_t_s],  # first 136
                                                                     matrix_size=self.data_tr.n_ag)
        ratio_matrix_work = get_rectangular_matrix_from_upper_triu(rvector=lhs_sample[u_t_s:2 * u_t_s],  # from 136-272
                                                                   matrix_size=self.data_tr.n_ag)
        ratio_matrix_other = get_rectangular_matrix_from_upper_triu(rvector=lhs_sample[2 * u_t_s:],  # from 272
                                                                    matrix_size=self.data_tr.n_ag)
        # Get modified contact matrices per layers
        cm_sim_school = (1 - ratio_matrix_school) * self.data_tr.data.contact_data["school"]
        cm_sim_work = (1 - ratio_matrix_work) * self.data_tr.data.contact_data["work"]
        cm_sim_other = (1 - ratio_matrix_other) * self.data_tr.data.contact_data["other"]
        # Get full contact matrix
        cm_sim = self.data_tr.contact_home + cm_sim_school + cm_sim_work + cm_sim_other
        # Get output from target calculator
        targ = TargetCalculator(data_tr=self.data_tr, cm_sim=cm_sim)
        output = targ.output
        cm_total_sim = (cm_sim * self.data_tr.age_vector)[self.data_tr.upper_tri_indexes]
        output = np.append(cm_total_sim, output)
        self.output = output
        return list(output)
