import numpy as np
import scipy.stats as ss

import src
from src.prcc.prcc import get_prcc_values, get_rectangular_matrix_from_upper_triu


class PRCCCalculator:
    """
    Calculate Partial Rank Correlation Coefficients (PRCC) and associated statistics
    for sensitivity analysis of epidemic simulations.
    """
    def __init__(self, sim_obj: src.SimulationNPI):
        self.sim_obj = sim_obj

        # Confidence intervals and p-values for aggregated PRCC
        self.confidence_lower = None
        self.confidence_upper = None
        self.p_value = None
        self.p_value_mtx = None
        self.prcc_mtx = None
        self.prcc_list = None
        self.agg_prcc = None  # Median PRCC per age group

    def calculate_prcc_values(self, lhs_table: np.ndarray, sim_output: np.ndarray):
        """
        Calculates PRCC values between LHS input parameters and simulation output.

        :param np.ndarray lhs_table: Latin Hypercube Sampling table of parameter sets.
        :param np.ndarray sim_output: Simulation outputs corresponding to each row in lhs_table.
        """
        # Combine LHS parameters and simulation output into a single matrix
        simulation = np.append(lhs_table, sim_output.reshape((-1, 1)), axis=1)

        # Compute the PRCC vector using the last column (output) against all parameters
        prcc_list = get_prcc_values(lhs_output_table=simulation)

        # Convert PRCC vector into a symmetric matrix for analysis
        self.prcc_mtx = get_rectangular_matrix_from_upper_triu(
            rvector=prcc_list[:self.sim_obj.upper_tri_size],  # only upper-triangular values
            matrix_size=self.sim_obj.n_ag
        )

        self.prcc_list = prcc_list

    def calculate_p_values(self):
        """
        Computes p-values for PRCC coefficients using Student's t-distribution.

        The t-statistic for each PRCC is calculated as:
            t = PRCC * sqrt((N - 2 - k) / (1 - PRCC^2))
        where N is the number of LHS samples and k is the number of parameters (age groups).
        """
        t = self.prcc_list * np.sqrt(
            (self.sim_obj.n_samples - 2 - self.sim_obj.n_ag) /
            (1 - (self.prcc_list ** 2))
        )

        # Degrees of freedom for two-sided t-test
        dof = self.sim_obj.n_samples - 2 - self.sim_obj.n_ag
        p_value = 2 * (1 - ss.t.cdf(x=np.abs(t), df=dof))

        # Store p-values in vector and convert to symmetric matrix
        self.p_value = p_value
        self.p_value_mtx = get_rectangular_matrix_from_upper_triu(
            rvector=p_value[:self.sim_obj.upper_tri_size],
            matrix_size=self.sim_obj.n_ag
        )

    def _calculate_distribution_prcc_p_val(self):
        """
        Calculates a normalized distribution of PRCC weights based on p-values.

        Each element represents the relative importance of that PRCC value
        in contributing to the overall sensitivity, normalized by row.
        """
        distribution_prcc_p_val = (1 - self.p_value_mtx) / \
                                  np.sum(1 - self.p_value_mtx, axis=1, keepdims=True)
        return distribution_prcc_p_val

    def aggregate_prcc_values_median(self):
        """
        Computes the median PRCC for each age group along with first and third quartiles.

        The aggregation is weighted using the normalized p-value distribution.
        """
        median_values = []
        conf_lower = []
        conf_upper = []

        # Compute normalized weights from p-values
        distribution_prcc_p_val = self._calculate_distribution_prcc_p_val()

        # Iterate over age groups (columns of PRCC matrix)
        for i in range(self.sim_obj.n_ag):
            prcc_column = self.prcc_mtx[:, i]
            prob_value_column = distribution_prcc_p_val[:, i]

            # Combine PRCC and probability weights for sorting
            combined_matrix = np.column_stack((np.abs(prcc_column), prob_value_column))

            # Sort rows by absolute PRCC values
            sorted_indices = np.argsort(combined_matrix[:, 0])
            combined_matrix_sorted = combined_matrix[sorted_indices]

            # Compute cumulative sum of probability weights
            cumul_distr = np.cumsum(combined_matrix_sorted[:, 1])

            # Determine median and quartiles based on cumulative probability
            median_value = combined_matrix_sorted[cumul_distr >= 0.5, 0][0]
            q1_value = combined_matrix_sorted[cumul_distr >= 0.25, 0][0]
            q3_value = combined_matrix_sorted[cumul_distr >= 0.75, 0][0]

            # Store aggregated PRCC and confidence intervals
            median_values.append(median_value)
            conf_lower.append(median_value - q1_value)
            conf_upper.append(q3_value - median_value)

        self.agg_prcc = median_values
        self.confidence_lower = conf_lower
        self.confidence_upper = conf_upper
