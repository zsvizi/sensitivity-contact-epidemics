from time import sleep

import numpy as np
from tqdm import tqdm

from prcc import get_contact_matrix_from_upper_triu, get_rectangular_matrix_from_upper_triu
from sampler_base import SamplerBase


class NPISampler(SamplerBase):
    def __init__(self, sim_state: dict, sim_obj):
        super().__init__(sim_state, sim_obj)
        self.susc = sim_state["susc"]
        # Matrices of frequently used contact types
        self.contact_home = self.sim_obj.contact_home
        self.contact_total = self.sim_obj.contact_matrix
        # Get number of elements in the upper triangular matrix
        self.upper_tri_size = int((self.sim_obj.no_ag + 1) * self.sim_obj.no_ag / 2)

        # Local variable for calculating boundaries
        lower_bound_mitigation = \
            self.contact_total * self.sim_obj.age_vector - \
            np.min((self.contact_total - self.contact_home) * self.sim_obj.age_vector)
        cm_total_symmetric = self.contact_total * self.sim_obj.age_vector

        self.lhs_boundaries = \
            {
             # Contact matrix entry level approach, full scale approach (old name: "home")
             "lockdown": {"lower": np.zeros(self.upper_tri_size),
                          "upper": np.ones(self.upper_tri_size)},
             # Contact matrix entry level approach, perturbation-like (old name: "normed")
             "mitigation": {"lower": lower_bound_mitigation[self.sim_obj.upper_tri_indexes],
                            "upper": cm_total_symmetric[self.sim_obj.upper_tri_indexes]},
             # Contact matrix entry level approach, full scale approach (old name: "home")
             "lockdown_3": {"lower": np.zeros(3 * self.upper_tri_size),
                            "upper": np.ones(3 * self.upper_tri_size)}
             }

    def run(self):
        # check if r0_lhs contains < 1
        print("computing kappa for base_r0=" + str(self.base_r0))
        r0_lhs_home = self._get_output(cm_sim=self.sim_obj.contact_home)
        kappa = None
        if r0_lhs_home[1] < 1:
            kappas = np.linspace(0, 1, 10_0)
            r0_home_kappas = np.array(list(map(self.kappify, kappas)))
            k = np.argmax(r0_home_kappas > 1)
            kappa = kappas[k]
            print(kappa)
            cm_diff = self.sim_obj.contact_matrix - self.sim_obj.contact_home
            cm_diff = cm_diff[np.triu_indices(self.sim_obj.no_ag)]

        # Get LHS table
        if self.type in ["lockdown_3"]:
            lhs_table = self._get_lhs_table(number_of_samples=120_000)
        else:
            lhs_table = self._get_lhs_table(number_of_samples=10_0, kappa=kappa, cm_diff=cm_diff)
        sleep(0.3)

        # Select getter for simulation output
        if self.type in ["lockdown"]:
            get_sim_output = self._get_sim_output_cm_entries_lockdown
        elif self.type in ["lockdown_3"]:
            get_sim_output = self._get_sim_output_cm_entries_lockdown_3
        elif self.type in ["mitigation"]:
            get_sim_output = self._get_sim_output_cm_entries
        else:
            raise Exception('Matrix type is unknown!')

        # Generate LHS output
        results = list(tqdm(map(get_sim_output, lhs_table), total=lhs_table.shape[0]))
        results = np.array(results)

        # check if all r0s are > 1
        res_min = results[:, 137].min()
        if res_min < 1:
            print("minimal lhs_r0: " + str(res_min))

        # Sort tables by R0 values
        r0_col_idx = int(self.upper_tri_size + 1)
        sorted_idx = results[:, r0_col_idx].argsort()
        results = results[sorted_idx]
        lhs_table = np.array(lhs_table[sorted_idx])
        sim_output = np.array(results)
        sleep(0.3)

        icu_maxes = list(tqdm(map(self.gen_icu_max, lhs_table), total=lhs_table.shape[0]))
        sim_output[:, 136] = icu_maxes

        # Save outputs
        self._save_output(output=lhs_table, folder_name='lhs')
        self._save_output(output=sim_output, folder_name='simulations')

    def gen_icu_max(self, line):
        time = np.arange(0, 1000, 0.5)
        cm = get_rectangular_matrix_from_upper_triu(line, self.sim_obj.no_ag)
        solution = self.sim_obj.model.get_solution(t=time, parameters=self.sim_obj.params, cm=cm)
        icu_max = solution[:, self.sim_obj.model.c_idx["ic"] *
                              self.sim_obj.no_ag:(self.sim_obj.model.c_idx["ic"] + 1) * self.sim_obj.no_ag].max()
        return icu_max

    def kappify(self, kappa):
        cm_sim = self.sim_obj.contact_home + kappa
        r0_lhs_home_k = self._get_output(cm_sim=cm_sim)
        return r0_lhs_home_k[1]

    def _get_variable_parameters(self):
        return [str(self.susc), str(self.base_r0), format(self.beta, '.5f'), self.type]

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

    def _get_sim_output_cm_entries_lockdown(self, lhs_sample: np.ndarray):
        # Get ratio matrix
        ratio_matrix = get_rectangular_matrix_from_upper_triu(rvector=lhs_sample,
                                                              matrix_size=self.sim_obj.no_ag)
        # Get modified full contact matrix
        cm_sim = (1 - ratio_matrix) * (self.sim_obj.contact_matrix - self.sim_obj.contact_home)
        cm_sim += self.sim_obj.contact_home
        # Get output
        output = self._get_output(cm_sim=cm_sim)
        cm_total_sim = (cm_sim * self.sim_obj.age_vector)[self.sim_obj.upper_tri_indexes]
        output = np.append(cm_total_sim, output)
        return list(output)

    def _get_sim_output_cm_entries_lockdown_3(self, lhs_sample: np.ndarray):
        # Get number of elements in the upper triangular matrix
        u_t_s = self.upper_tri_size
        # Get ratio matrices
        ratio_matrix_school = get_rectangular_matrix_from_upper_triu(rvector=lhs_sample[:u_t_s],
                                                                     matrix_size=self.sim_obj.no_ag)
        ratio_matrix_work = get_rectangular_matrix_from_upper_triu(rvector=lhs_sample[u_t_s:2*u_t_s],
                                                                   matrix_size=self.sim_obj.no_ag)
        ratio_matrix_other = get_rectangular_matrix_from_upper_triu(rvector=lhs_sample[2*u_t_s:],
                                                                    matrix_size=self.sim_obj.no_ag)
        # Get modified contact matrices per layers
        cm_sim_school = (1 - ratio_matrix_school) * self.sim_obj.data.contact_data["school"]
        cm_sim_work = (1 - ratio_matrix_work) * self.sim_obj.data.contact_data["work"]
        cm_sim_other = (1 - ratio_matrix_other) * self.sim_obj.data.contact_data["other"]
        # Get full contact matrix
        cm_sim = self.sim_obj.contact_home + cm_sim_school + cm_sim_work + cm_sim_other
        # Get output
        output = self._get_output(cm_sim=cm_sim)
        cm_total_sim = (cm_sim * self.sim_obj.age_vector)[self.sim_obj.upper_tri_indexes]
        output = np.append(cm_total_sim, output)
        return list(output)

    def _get_upper_bound_factor_unit(self):
        cm_diff = (self.sim_obj.contact_matrix - self.contact_home) * self.sim_obj.age_vector
        min_diff = np.min(cm_diff) / 2
        return min_diff
