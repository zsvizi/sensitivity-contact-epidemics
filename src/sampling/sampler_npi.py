import os
from time import sleep

import numpy as np
from tqdm import tqdm

import src
from src.sampling.cm_calculator_lockdown import CMCalculatorLockdown
from src.sampling.cm_calculator_lockdown_typewise import CMCalculatorLockdownTypewise
from src.prcc import get_rectangular_matrix_from_upper_triu
from src.sampling.sampler_base import SamplerBase
from src.sampling.target_calculator import TargetCalculator


class SamplerNPI(SamplerBase):
    def __init__(self, sim_state: dict, sim_obj: src.SimulationNPI,
                 mtx_type: str = "lockdown") -> None:
        super().__init__(sim_state=sim_state, sim_obj=sim_obj)
        self.sim_obj = sim_obj

        if mtx_type == "lockdown":
            cm_calc = CMCalculatorLockdown(sim_obj=self.sim_obj, sim_state=sim_state)
            self.get_sim_output = cm_calc.get_sim_output_cm_entries_lockdown
        elif mtx_type == "lockdown_3":
            cm_calc = CMCalculatorLockdownTypewise(sim_obj=self.sim_obj, sim_state=sim_state)
            self.get_sim_output = cm_calc.get_sim_output_cm_entries_lockdown_3
        else:
            raise Exception("Matrix type is unknown!")

        self.susc = sim_state["susc"]

        # Matrices of frequently used contact types
        self.contact_home = self.sim_obj.contact_home
        self.contact_total = self.sim_obj.contact_matrix

        self.upper_tri_size = int((self.sim_obj.n_ag + 1) * self.sim_obj.n_ag / 2)

        self.lhs_boundaries = cm_calc.lhs_boundaries

    def run(self):
        kappa = self.calculate_kappa()
        # check if r0_lhs contains < 1
        print("computing kappa for base_r0=" + str(self.base_r0))
        # get output from target calculator
        tar_out = TargetCalculator(sim_obj=self.sim_obj, sim_state=self.sim_state)
        r0_lhs_home = tar_out.get_output(cm_sim=self.contact_home)
        if r0_lhs_home[1] < 1:
            # Get LHS table
            number_of_samples = 120000
            lhs_table = self._get_lhs_table(number_of_samples=number_of_samples, kappa=kappa)

            # Results have shape of (number_of_samples, 136 + 1 + 1 + 16)
            results = list(tqdm(map(self.get_sim_output, lhs_table), total=lhs_table.shape[0]))
            results = np.array(results)

            # check if all r0s are > 1
            r0_col_idx = int(self.upper_tri_size - 1 + 2)
            res_min = results[:, r0_col_idx].min()
            if res_min < 1:
                print("minimal lhs_r0: " + str(res_min))

            # Sort tables by R0 values
            sorted_idx = results[:, r0_col_idx].argsort()
            results = results[sorted_idx]
            lhs_table = np.array(lhs_table[sorted_idx])
            sim_output = np.array(results)
            sleep(0.3)

            # Save outputs
            self._save_output(output=lhs_table, folder_name='lhs')  # (12, 000, 136)
            self._save_output(output=sim_output, folder_name='simulations')  # (12, 000, 136)
            return lhs_table, sim_output

    def gen_icu_max(self, line) -> np.ndarray:
        time = np.arange(0, 1000, 0.5)
        cm = get_rectangular_matrix_from_upper_triu(line, self.sim_obj.n_ag)
        solution = self.sim_obj.model.get_solution(t=time, parameters=self.sim_obj.params, cm=cm)
        idx_icu = self.sim_obj.model.c_idx["ic"] * self.sim_obj.n_ag
        icu_max = solution[:, idx_icu:(idx_icu + self.sim_obj.n_ag)].max()
        return icu_max

    def calculate_kappa(self):
        kappas = np.linspace(0, 1, 1000)
        r0_home_kappas = np.array(list(map(self.kappify, kappas)))
        k = np.argmax(r0_home_kappas > 1, axis=0)  # Returns the indices of the maximum values along an axis.
        kappa = kappas[k]
        print("k", kappa)
        return kappa

    def kappify(self, kappa: float = None) -> float:
        os.makedirs("./sens_data/cm", exist_ok=True)
        cm_diff = self.sim_obj.contact_matrix - self.sim_obj.contact_home
        cm_sim = self.sim_obj.contact_home + kappa * cm_diff
        # save cm to use in aggregation
        np.savetxt("sens_data/cm/cm.csv", X=cm_sim, delimiter=";")
        # get output from target calculator
        tar_out = TargetCalculator(sim_obj=self.sim_obj, sim_state=self.sim_state)
        r0_lhs_home_k = tar_out.get_output(cm_sim=cm_sim)
        return r0_lhs_home_k[1]

    def _get_variable_parameters(self):
        return [str(self.susc), str(self.base_r0), format(self.beta, '.5f'), self.type]

    def _get_upper_bound_factor_unit(self) -> np.ndarray:
        cm_diff = (self.sim_obj.contact_matrix - self.contact_home) * self.sim_obj.age_vector
        min_diff = np.min(cm_diff) / 2
        return min_diff
