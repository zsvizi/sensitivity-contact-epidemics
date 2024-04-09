import numpy as np
import scipy.stats as ss

import src
from src.prcc import get_prcc_values, get_rectangular_matrix_from_upper_triu


class PRCCCalculator:
    def __init__(self, sim_obj: src.SimulationNPI):
        self.sim_obj = sim_obj

        self.p_value = None
        self.p_value_mtx = None
        self.prcc_mtx = None
        self.prcc_list = None
        self.agg_prcc = None
        self.agg_std = None

    def calculate_prcc_values(self, lhs_table: np.ndarray, sim_output: np.ndarray):
        sim_data = lhs_table[:, :(self.sim_obj.n_ag * (self.sim_obj.n_ag + 1)) // 2]
        sim_data = 1 - sim_data
        simulation = np.append(sim_data, sim_output.reshape((-1, 1)), axis=1)
        prcc_list = get_prcc_values(lhs_output_table=simulation)
        prcc_mtx = get_rectangular_matrix_from_upper_triu(
            rvector=prcc_list[:self.sim_obj.upper_tri_size],
            matrix_size=self.sim_obj.n_ag)
        self.prcc_mtx = prcc_mtx
        self.prcc_list = prcc_list

    def calculate_p_values(self):
        t = self.prcc_list * np.sqrt(
            (self.sim_obj.n_samples - 2 - self.sim_obj.upper_tri_size) / (1 - self.prcc_list ** 2)
        )
        # p-value for 2-sided test
        dof = self.sim_obj.n_samples - 2 - self.sim_obj.upper_tri_size
        p_value = 2 * (1 - ss.t.cdf(x=abs(t), df=dof))
        self.p_value = p_value
        self.p_value_mtx = get_rectangular_matrix_from_upper_triu(
            rvector=p_value[:self.sim_obj.upper_tri_size],
            matrix_size=self.sim_obj.n_ag)

    def aggregate_prcc_values(self):
        distribution_p_val = (
                (1 - self.p_value_mtx) /
                np.sum(1 - self.p_value_mtx, axis=1, keepdims=True)
        )
        use_complex_logic = False
        if use_complex_logic:
            distribution_prcc_p_val = (
                    (self.prcc_mtx * distribution_p_val) /
                    np.sum(self.prcc_mtx * distribution_p_val, axis=1, keepdims=True)
            )
        else:
            distribution_prcc_p_val = distribution_p_val.copy()

        agg = np.sum(
            self.prcc_mtx * distribution_prcc_p_val,
            axis=1
        )

        agg_square = np.sum(
            self.prcc_mtx ** 2 * distribution_prcc_p_val,
            axis=1
        )
        agg_std = np.sqrt(agg_square - agg ** 2)

        self.agg_std = agg_std
        self.agg_prcc = agg
        return agg.flatten(), agg_std.flatten()
