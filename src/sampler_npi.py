from time import sleep

import numpy as np
from tqdm import tqdm

from src.prcc import get_rectangular_matrix_from_upper_triu
from src.sampler_base import SamplerBase
from src.CMCalculatorLockdown import Lockdown
from src.CMCalculatorLockdownTypewise import Lockdown3
from src.target_calculation import TargetCalculator
from src.data_transformer import Transformer


class SamplerNPI(SamplerBase):
    def __init__(self, sim_state: dict, sim_obj: Transformer,
                 result: str = "lockdown") -> None:
        self.sim_state = sim_state
        self.sim_obj = sim_obj
        if result == "lockdown":
            result = Lockdown(sim_obj=self.sim_obj)
            print(result)
        elif result == "lockdown3":
            result = Lockdown3(sim_obj=self.sim_obj)
            self.result = result.output

        super().__init__(sim_state, sim_obj=sim_obj)
        self.susc = sim_state["susc"]

        self.sim_obj = Transformer()

        # Matrices of frequently used contact types
        self.contact_home = self.sim_obj.contact_home
        self.contact_total = self.sim_obj.contact_matrix

        # Local variable for calculating boundaries
        lower_bound_mitigation = \
            self.contact_total * self.sim_obj.age_vector - \
            np.min((self.contact_total - self.contact_home) * self.sim_obj.age_vector)
        cm_total_symmetric = self.contact_total * self.sim_obj.age_vector

        self.lhs_boundaries = \
            {
                # Contact matrix entry level approach, perturbation-like (old name: "normed")
                "mitigation": {"lower": lower_bound_mitigation[self.sim_obj.upper_tri_size],
                               "upper": cm_total_symmetric[self.sim_obj.upper_tri_size]}
            }

    def run(self):
        maxiter = 120_000
        # check if r0_lhs contains < 1
        print("computing kappa for base_r0=" + str(self.base_r0))
        # get output from target calculator
        tar_out = TargetCalculator(sim_obj=self.sim_obj)
        r0_lhs_home = tar_out.output(cm_sim=self.sim_obj.contact_home)

        # r0_lhs_home = self._get_output(cm_sim=self.sim_obj.contact_home)
        kappa = None
        if r0_lhs_home[1] < 1:
            kappas = np.linspace(0, 1, 10_000)
            r0_home_kappas = np.array(list(map(self.kappify, kappas)))
            k = np.argmax(r0_home_kappas > 1)
            kappa = kappas[k]
            print(kappa)

        # Get LHS table
            # Get LHS table
        lhs_table = self._get_lhs_table(number_of_samples=maxiter, kappa=kappa, sim_obj=self.sim_obj)
        sleep(0.3)

        # Select getter for simulation output
        get_sim_output = self.result

        # Generate LHS output
        results = list(tqdm(map(get_sim_output, lhs_table), total=lhs_table.shape[0]))
        results = np.array(results)

        # check if all r0s are > 1
        res_min = results[:, 137].min()
        if res_min < 1:
            print("minimal lhs_r0: " + str(res_min))

        # Sort tables by R0 values
        r0_col_idx = int(self.sim_obj.upper_tri_size + 1)
        sorted_idx = results[:, r0_col_idx].argsort()
        results = results[sorted_idx]
        lhs_table = np.array(lhs_table[sorted_idx])
        sim_output = np.array(results)
        sleep(0.3)

        # Save outputs
        self._save_output(output=lhs_table, folder_name='lhs')
        self._save_output(output=sim_output, folder_name='simulations')

    def gen_icu_max(self, line) -> np.ndarray:
        time = np.arange(0, 1000, 0.5)
        cm = get_rectangular_matrix_from_upper_triu(line, self.sim_obj.n_ag)
        solution = self.sim_obj.model.get_solution(t=time, parameters=self.sim_obj.params, cm=cm)
        icu_max = solution[:, self.sim_obj.model.c_idx["ic"] * self.sim_obj.n_ag:(self.sim_obj.model.c_idx["ic"] + 1) *
                           self.sim_obj.n_ag].max()
        return icu_max

    def kappify(self, kappa) -> float:
        cm_diff = self.sim_obj.contact_matrix - self.sim_obj.contact_home
        cm_sim = self.sim_obj.contact_home + kappa * cm_diff

        # get output from target calculator
        tar_out = TargetCalculator(sim_obj=self.sim_obj)
        r0_lhs_home_k = tar_out.output(cm_sim=cm_sim)
        return r0_lhs_home_k[1]

    def _get_variable_parameters(self):
        return [str(self.susc), str(self.base_r0), format(self.beta, '.5f'), self.type]

    def _get_upper_bound_factor_unit(self) -> np.ndarray:
        cm_diff = (self.sim_obj.contact_matrix - self.contact_home) * self.sim_obj.age_vector
        min_diff = np.min(cm_diff) / 2
        return min_diff
