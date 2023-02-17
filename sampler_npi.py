from time import sleep

import numpy as np
from tqdm import tqdm

from prcc import get_contact_matrix_from_upper_triu, get_rectangular_matrix_from_upper_triu
from sampler_base import SamplerBase
from lockdown import Lockdown
from lockdown3 import Lockdown3
from mitigation import Mitigation
from Target_calculation import TargetCalculator


class NPISampler(SamplerBase):
    def __init__(self, sim_state: dict, sim_obj, get_output: TargetCalculator,
                 cm_entries: Mitigation, entries_lockdown: Lockdown, entries_lockdown3: Lockdown3):
        super().__init__(sim_state, sim_obj)
        self.susc = sim_state["susc"]
        self.get_output = get_output
        self.get_sim_output_cm_entries = cm_entries
        self.get_sim_output_cm_entries_lockdown = entries_lockdown
        self.get_sim_output_cm_entries_lockdown_3 = entries_lockdown3

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
        maxiter = 120_000
        # check if r0_lhs contains < 1
        print("computing kappa for base_r0=" + str(self.base_r0))
        r0_lhs_home = TargetCalculator.get_output(self=self.sim_obj, cm_sim=self.sim_obj.contact_home)
        kappa = None
        if r0_lhs_home[1] < 1:
            kappas = np.linspace(0, 1, 10_000)
            r0_home_kappas = np.array(list(map(self.kappify, kappas)))
            k = np.argmax(r0_home_kappas > 1)
            kappa = kappas[k]
            print(kappa)

        # Get LHS table
        if self.type in ["lockdown_3"]:
            lhs_table = self._get_lhs_table(number_of_samples=120_000)
        else:
            lhs_table = self._get_lhs_table(number_of_samples=maxiter, kappa=kappa, sim_obj=self.sim_obj)
        sleep(0.3)
        # Select getter for simulation output
        if self.type in ["lockdown"]:
            get_sim_output = self.get_sim_output_cm_entries_lockdown
        elif self.type in ["lockdown_3"]:
            get_sim_output = self.get_sim_output_cm_entries_lockdown_3
        elif self.type in ["mitigation"]:
            get_sim_output = self.get_sim_output_cm_entries
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

        # icu_maxes = list(tqdm(map(self.gen_icu_max, lhs_table), total=lhs_table.shape[0]))
        # sim_output[:, 136] = icu_maxes

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
        cm_diff = self.sim_obj.contact_matrix - self.sim_obj.contact_home
        cm_sim = self.sim_obj.contact_home + kappa * cm_diff
        r0_lhs_home_k = TargetCalculator.get_output(self=self.sim_obj, cm_sim=cm_sim)
        return r0_lhs_home_k[1]

    def _get_variable_parameters(self):
        return [str(self.susc), str(self.base_r0), format(self.beta, '.5f'), self.type]

    def _get_upper_bound_factor_unit(self):
        cm_diff = (self.sim_obj.contact_matrix - self.contact_home) * self.sim_obj.age_vector
        min_diff = np.min(cm_diff) / 2
        return min_diff

