from abc import ABC, abstractmethod
import os
from time import sleep

import numpy as np
from smt.sampling_methods import LHS
from tqdm import tqdm

from prcc import get_contact_matrix_from_upper_triu


class SamplerBase(ABC):
    def __init__(self, sim_state: dict, sim_obj):
        self.sim_obj = sim_obj
        self.base_r0 = sim_state["base_r0"]
        self.beta = sim_state["beta"]
        self.mtx_type = sim_state["mtx_type"]
        self.susc = sim_state["susc"]
        self.r0generator = sim_state["r0generator"]
        self.lhs_boundaries = None

    @abstractmethod
    def run(self):
        pass

    def _get_lhs_table(self):
        # Get actual limit matrices
        lower_bound = self.lhs_boundaries[self.mtx_type]["lower"]
        upper_bound = self.lhs_boundaries[self.mtx_type]["upper"]
        # Get LHS tables
        number_of_samples = 40000
        lhs_table = create_latin_table(n_of_samples=number_of_samples,
                                       lower=lower_bound,
                                       upper=upper_bound)
        print("Simulation for", number_of_samples,
              "samples (", "-".join([str(self.susc), str(self.base_r0), self.mtx_type]), ")")
        return lhs_table

    @abstractmethod
    def _get_output(self, cm_sim: np.ndarray):
        pass

    def _save_output(self, output, folder_name):
        # Create directories for saving calculation outputs
        os.makedirs("./sens_data", exist_ok=True)

        # Save LHS output
        os.makedirs("./sens_data/" + folder_name, exist_ok=True)
        filename = "./sens_data/" + folder_name + "/" + folder_name + "_Hungary_" + \
                   "_".join([str(self.susc), str(self.base_r0), format(self.beta, '.5f'), self.mtx_type])
        np.savetxt(fname=filename + ".csv", X=np.asarray(output), delimiter=";")


class ContactMatrixSampler(SamplerBase):
    def __init__(self, sim_state: dict, sim_obj):
        super().__init__(sim_state, sim_obj)
        # Matrices of frequently used contact types
        self.contact_home = self.sim_obj.contact_home
        self.contact_total = self.sim_obj.contact_matrix

        # Local variable for calculating boundaries
        lower_bound_mitigation = \
            self.contact_total * self.sim_obj.age_vector - \
            np.min((self.contact_total - self.contact_home) * self.sim_obj.age_vector)
        cm_home_symmetric = self.contact_home * self.sim_obj.age_vector
        cm_total_symmetric = self.contact_total * self.sim_obj.age_vector

        self.lhs_boundaries = \
            {
             # Age group level perturbation-like approach (NOT USED)
             "unit": {"lower": np.zeros(self.sim_obj.no_ag),
                      "upper": np.ones(self.sim_obj.no_ag) * self._get_upper_bound_factor_unit()},
             # Age group level full scale (it is possible to make zero) approach
             "ratio": {"lower": np.zeros(3 * self.sim_obj.no_ag),
                       "upper": 0.5 * np.ones(3 * self.sim_obj.no_ag)},
             # Contact matrix entry level approach, full scale approach (old name: "home")
             "lockdown": {"lower": cm_home_symmetric[self.sim_obj.upper_tri_indexes],
                          "upper": cm_total_symmetric[self.sim_obj.upper_tri_indexes]},
             # Contact matrix entry level approach, perturbation-like (old name: "normed")
             "mitigation": {"lower": lower_bound_mitigation[self.sim_obj.upper_tri_indexes],
                            "upper": cm_total_symmetric[self.sim_obj.upper_tri_indexes]}
             }

    def run(self):
        # Get LHS table
        lhs_table = self._get_lhs_table()
        sleep(0.3)

        # Select getter for simulation output
        if self.mtx_type == 'unit':
            get_sim_output = self._get_sim_output_unit
        elif self.mtx_type == 'ratio':
            get_sim_output = self._get_sim_output_ratio
        elif self.mtx_type == 'lockdown' or self.mtx_type == 'mitigation':
            get_sim_output = self._get_sim_output_cm_entries
        else:
            raise Exception('Matrix type is unknown!')

        # Generate LHS output
        results = list(tqdm(map(get_sim_output, lhs_table), total=lhs_table.shape[0]))
        results = np.array(results)

        # Sort tables by R0 values
        r0_col_idx = int((self.sim_obj.no_ag + 1) * self.sim_obj.no_ag / 2 + 1)
        sorted_idx = results[:, r0_col_idx].argsort()
        results = results[sorted_idx]
        lhs_table = np.array(lhs_table[sorted_idx])
        sim_output = np.array(results)
        sleep(0.3)

        # Save outputs
        self._save_output(output=lhs_table, folder_name='lhs')
        self._save_output(output=sim_output, folder_name='simulations')

    def _get_output(self, cm_sim: np.ndarray):
        beta_lhs = self.base_r0 / self.r0generator.get_eig_val(contact_mtx=cm_sim,
                                                               susceptibles=self.sim_obj.susceptibles.reshape(1, -1),
                                                               population=self.sim_obj.population)[0]
        r0_lhs = (self.beta / beta_lhs) * self.base_r0
        output = np.array([0, r0_lhs])
        output = np.append(output, np.zeros(self.sim_obj.no_ag))
        return output

    def _get_sim_output_cm_entries(self, lhs_sample: np.ndarray):
        # Get output
        cm_sim = get_contact_matrix_from_upper_triu(rvector=lhs_sample,
                                                    age_vector=self.sim_obj.age_vector.reshape(-1,))
        output = self._get_output(cm_sim=cm_sim)
        output = np.append(lhs_sample, output)
        return list(output)

    def _get_sim_output_unit(self, lhs_sample: np.ndarray):
        # Subtract lhs_sample as a column from cm_total_full (= reduction of row sum)
        cm_total_sim = self.sim_obj.contact_matrix * self.sim_obj.age_vector - lhs_sample.reshape(-1, 1)
        # Subtract lhs_sample as a row (reduction of col sum)
        cm_total_sim -= lhs_sample.reshape(1, -1)
        # Diagonal elements were reduced twice -> correction
        cm_total_sim += np.diag(lhs_sample)
        # Get output
        cm_sim = cm_total_sim / self.sim_obj.age_vector
        output = self._get_output(cm_sim=cm_sim)
        output = np.append(cm_total_sim[self.sim_obj.upper_tri_indexes], output)
        return list(output)

    def _get_sim_output_ratio(self, lhs_sample: np.ndarray):
        # Number of age groups
        no_ag = self.sim_obj.no_ag

        # Local function for calculating factor matrix for all contact types
        def get_factor(sampled_ratios):
            # get ratio matrix via multiplying 1-matrix by sampled_ratios as a column
            ratio_col = np.ones((no_ag, no_ag)) * sampled_ratios.reshape((-1, 1))
            # get ratio matrix via multiplying 1-matrix by sampled_ratios as a row
            ratio_row = np.ones((no_ag, no_ag)) * sampled_ratios.reshape((1, -1))
            # create factor matrix via adding up ratio_col and ratio_row
            # in order to get a factor for total matrix of a specific contact type, subtract the sum from 1
            factor_matrix = 1 - (ratio_col + ratio_row)
            return factor_matrix

        # Contact data from Simulation object
        contact_data = self.sim_obj.data.contact_data
        # Modified total contact matrices (contact types: school, work, other)
        cm_mod_school = get_factor(lhs_sample[:no_ag]) * (contact_data["school"] * self.sim_obj.age_vector)
        cm_mod_work = get_factor(lhs_sample[no_ag:2*no_ag]) * (contact_data["work"] * self.sim_obj.age_vector)
        cm_mod_other = get_factor(lhs_sample[2*no_ag:]) * (contact_data["other"] * self.sim_obj.age_vector)
        # Get modified total contact matrix of type full
        cm_total_home = contact_data["home"] * self.sim_obj.age_vector
        cm_total_sim = cm_total_home + cm_mod_school + cm_mod_work + cm_mod_other
        # Get output
        cm_sim = cm_total_sim / self.sim_obj.age_vector
        output = self._get_output(cm_sim=cm_sim)
        output = np.append(cm_total_sim[self.sim_obj.upper_tri_indexes], output)
        return list(output)

    def _get_upper_bound_factor_unit(self):
        cm_diff = (self.sim_obj.contact_matrix - self.contact_home) * self.sim_obj.age_vector
        min_diff = np.min(cm_diff) / 2
        return min_diff


def create_latin_table(n_of_samples, lower, upper):
    bounds = np.array([lower, upper]).T
    sampling = LHS(xlimits=bounds)
    return sampling(n_of_samples)
