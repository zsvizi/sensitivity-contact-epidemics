import numpy as np
import scipy.stats as ss

import src
from src.prcc import get_prcc_values, get_rectangular_matrix_from_upper_triu


class PRCCCalculator:
    def __init__(self, sim_obj: src.SimulationNPI):
        self.std_upper = None
        self.std_lower = None
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

    def _calculate_distribution_prcc_p_val(self, prcc_mtx, p_value_mtx):
        distribution_p_val = prcc_mtx * (1 - p_value_mtx)
        complex_logic = True
        if complex_logic:
            distribution_prcc_p_val = distribution_p_val / \
                                      np.sum(prcc_mtx * (1 - p_value_mtx), axis=1,
                                             keepdims=True)
        else:
            distribution_prcc_p_val = distribution_p_val.copy()
        return distribution_prcc_p_val

    def _calculate_aggregate_mean_std(self, prcc_mtx, distribution_prcc_p_val):
        agg = np.sum(prcc_mtx * distribution_prcc_p_val, axis=1)
        agg_square = np.sum(prcc_mtx ** 2 * distribution_prcc_p_val, axis=1)
        # Ensure that the difference is non-negative before taking the square root
        agg_std = np.sqrt(np.maximum(0, agg_square - agg ** 2))
        return agg, agg_std

    def aggregate_prcc_values_mean(self):
        distribution_prcc_p_val = self._calculate_distribution_prcc_p_val(self.prcc_mtx,
                                                                          self.p_value_mtx)
        agg = np.sum(self.prcc_mtx * distribution_prcc_p_val, axis=1)
        agg_square = np.sum(self.prcc_mtx ** 2 * distribution_prcc_p_val, axis=1)
        agg_std = np.sqrt(agg_square - agg ** 2)

        self.agg_std = agg_std
        self.agg_prcc = agg
        return agg.flatten(), agg_std.flatten()

    def aggregate_prcc_values_log(self):
        prcc_mtx_log = np.log(np.abs(self.prcc_mtx))
        distribution_prcc_p_val = self._calculate_distribution_prcc_p_val(prcc_mtx_log,
                                                                          self.p_value_mtx)
        agg, agg_std = self._calculate_aggregate_mean_std(prcc_mtx_log,
                                                          distribution_prcc_p_val)
        agg_std_lower = agg - agg_std
        agg_std_upper = agg + agg_std

        # transform back
        agg = np.exp(agg)
        agg_std = np.exp(agg_std)
        agg_std_lower = np.exp(agg_std_lower)
        agg_std_upper = np.exp(agg_std_upper)

        self.std_lower = agg_std_lower
        self.std_upper = agg_std_upper
        self.agg_std = agg_std
        self.agg_prcc = agg

        return agg.flatten(), agg_std_lower.flatten(), agg_std_upper.flatten()

    def aggregate_prcc_values_median(self):
        distribution_prcc_p_val = self._calculate_distribution_prcc_p_val(self.prcc_mtx,
                                                                          self.p_value_mtx)

        sort_indices = np.argsort(distribution_prcc_p_val[:, 0])
        distribution_prcc_p_val_sorted = distribution_prcc_p_val[sort_indices]
        sorted_values = np.sort(self.prcc_mtx[sort_indices], axis=1)

        cumulative_probs = np.cumsum(distribution_prcc_p_val_sorted, axis=1)

        median_index = np.argmax(cumulative_probs >= 0.5, axis=1)
        median_row = np.array([sorted_values[i][idx] for i,
                                                         idx in enumerate(median_index)])

        q1_index = np.argmax(cumulative_probs >= 0.25, axis=1)
        q3_index = np.argmax(cumulative_probs >= 0.75, axis=1)
        q1 = np.array([sorted_values[i][idx] for i, idx in enumerate(q1_index)])
        q3 = np.array([sorted_values[i][idx] for i, idx in enumerate(q3_index)])

        iqr = q3 - q1
        agg_std = iqr / 1.35

        self.std_lower = q1
        self.std_upper = q3
        self.agg_prcc = median_row
        self.agg_std = agg_std

        return self.agg_prcc.flatten(), q1.flatten(), q3.flatten()

    def aggregate_prcc_values(self, calculation_approach: str):
        if calculation_approach == 'mean':
            return self.aggregate_prcc_values_mean()
        elif calculation_approach == 'log':
            return self.aggregate_prcc_values_log()
        else:
            return self.aggregate_prcc_values_median()
