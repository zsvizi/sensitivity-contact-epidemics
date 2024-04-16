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

    def aggregate_prcc_values_median(self):
        median_values = []
        q1_values = []
        q3_values = []
        std_values = []
        # prob using complex logic
        distribution_prcc_p_val = self._calculate_distribution_prcc_p_val(self.prcc_mtx,
                                                                          self.p_value_mtx)

        # Iterate over the columns of prcc and p_value
        for i in range(16):
            # Take the ith column from prcc
            prcc_column = self.prcc_mtx[:, i]

            # Take the ith column from distribution_prcc_p_val
            prob_value_column = distribution_prcc_p_val[:, i]

            # Combine prcc_column and prob_value_column into a single matrix (16 * 2)
            combined_matrix = np.column_stack((prcc_column, prob_value_column))
            # Take the absolute values of the first column to avoid -ve median values
            combined_matrix[:, 0] = np.abs(combined_matrix[:, 0])
            # Sort the rows of combined_matrix by the first column
            sorted_indices = np.argsort(combined_matrix[:, 0])
            combined_matrix_sorted = combined_matrix[sorted_indices]
            # Calculate the cumulative sum of the second column
            second_column_cumsum = np.cumsum(combined_matrix_sorted[:, 1])
            # Find the index where the cumulative sum >= 0.5 for median
            median_index = np.argmax(second_column_cumsum >= 0.5)
            median_value = combined_matrix_sorted[median_index, 0]

            # Find the index where the cumulative sum >= 0.25 for Q1
            q1_indices = np.where(second_column_cumsum >= 0.25)[0]
            q1_index = q1_indices[0] if len(q1_indices) > 0 else median_index
            q1_value = combined_matrix_sorted[q1_index, 0]

            # Find the index where the cumulative sum >= 0.75 for Q3
            q3_index = np.where(second_column_cumsum >= 0.75)[0]
            q3_index = q3_index[-1] if len(q3_index) > 0 else median_index
            q3_value = combined_matrix_sorted[q3_index, 0]
            # calculate std dev
            std_dev = (q3_value - q1_value) / 1.35

            # Append the median, Q1, and Q3 values to their respective lists
            median_values.append(median_value)
            q1_values.append(q1_value)
            q3_values.append(q3_value)
            std_values.append(std_dev)

        self.agg_prcc = median_values
        self.agg_std = std_values
        self.std_lower = q1_values
        self.std_upper = q3_values

        return median_values, std_values

    def aggregate_prcc_values(self, calculation_approach: str):
        if calculation_approach == 'mean':
            return self.aggregate_prcc_values_mean()
        else:
            return self.aggregate_prcc_values_median()
