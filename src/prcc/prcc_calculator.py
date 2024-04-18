import numpy as np
import scipy.stats as ss

import src
from src.prcc.prcc import get_prcc_values, get_rectangular_matrix_from_upper_triu


class PRCCCalculator:
    def __init__(self, sim_obj: src.SimulationNPI, calculation_approach):
        self.sim_obj = sim_obj
        self.calculation_approach = calculation_approach

        self.agg_prcc = None
        self.confidence_lower = None
        self.confidence_upper = None
        self.p_value = None
        self.p_value_mtx = None
        self.prcc_mtx = None
        self.prcc_list = None
        self.agg_std = None

        self.complex_logic = False
        self.pop_logic = True

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

    def _calculate_distribution_prcc_p_val(self):
        distribution_p_val = self.prcc_mtx * (1 - self.p_value_mtx)
        if self.complex_logic:
            distribution_prcc_p_val = distribution_p_val / \
                                      np.sum(self.prcc_mtx * (1 - self.p_value_mtx), axis=1,
                                             keepdims=True)
        elif self.pop_logic:
            distribution_prcc_p_val = ((1 - self.p_value_mtx) *
                                       self.sim_obj.population.reshape((1, -1)) /
                                       np.sum((1 - self.p_value_mtx) *
                                              self.sim_obj.population.reshape((1, -1)),
                                              axis=1, keepdims=True))
        else:
            distribution_prcc_p_val = (1 - self.p_value_mtx) / np.sum(1 - self.p_value_mtx,
                                                                      axis=1, keepdims=True)
        return distribution_prcc_p_val

    def aggregate_prcc_values_mean(self):
        distribution_prcc_p_val = self._calculate_distribution_prcc_p_val()
        agg = np.sum(self.prcc_mtx * distribution_prcc_p_val, axis=1)
        agg_square = np.sum(self.prcc_mtx ** 2 * distribution_prcc_p_val, axis=1)
        agg_std = np.sqrt(agg_square - agg ** 2)

        self.confidence_lower = agg_std
        self.confidence_upper = agg_std
        self.agg_prcc = agg
        return agg.flatten(), agg_std.flatten()

    def aggregate_prcc_values_median(self):
        median_values = []
        conf_lower = []
        conf_upper = []
        # prob using complex logic
        distribution_prcc_p_val = self._calculate_distribution_prcc_p_val()

        # Iterate over the columns of prcc and p_value
        for i in range(16):
            # Take the ith column from prcc
            prcc_column = self.prcc_mtx[i, :]

            # Take the ith column from distribution_prcc_p_val
            prob_value_column = distribution_prcc_p_val[i, :]

            # Combine prcc_column and prob_value_column into a single matrix (16 * 2)
            combined_matrix = np.column_stack((prcc_column, prob_value_column))
            # Take the absolute values of the first column to avoid -ve median values
            combined_matrix[:, 0] = np.abs(combined_matrix[:, 0])
            # Sort the rows of combined_matrix by the first column
            sorted_indices = np.argsort(combined_matrix[:, 0])
            combined_matrix_sorted = combined_matrix[sorted_indices]
            # Calculate the cumulative sum of the second column
            cumul_distr = np.cumsum(combined_matrix_sorted[:, 1])
            # Find the median value
            median_value = combined_matrix_sorted[cumul_distr >= 0.5, 0][0]
            # Find the first quartile
            q1_value = combined_matrix_sorted[cumul_distr >= 0.25, 0][0]
            # Find the third quartile
            q3_value = combined_matrix_sorted[cumul_distr >= 0.75, 0][0]

            # Append the median, Q1, and Q3 values to their respective lists
            median_values.append(median_value)
            conf_lower.append(median_value - q1_value)
            conf_upper.append(q3_value - median_value)

        self.agg_prcc = median_values
        self.confidence_lower = conf_lower
        self.confidence_upper = conf_upper

    def aggregate_prcc_values(self):
        if self.calculation_approach == 'mean':
            return self.aggregate_prcc_values_mean()
        else:
            return self.aggregate_prcc_values_median()
