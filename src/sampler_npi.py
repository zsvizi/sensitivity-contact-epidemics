from time import sleep

import numpy as np
from tqdm import tqdm

from src.simulation_base import SimulationBase
from src.cm_calculator_lockdown import CMCalculatorLockdown
from src.cm_calculator_lockdown_typewise import CMCalculatorLockdownTypewise
from src.prcc import get_rectangular_matrix_from_upper_triu
from src.sampler_base import SamplerBase
from src.target_calculation import TargetCalculator


class SamplerNPI(SamplerBase):
    def __init__(self, sim_state: dict, data_tr: SimulationBase,
                 mtx_type: str = "lockdown") -> None:
        self.sim_state = sim_state
        self.data_tr = data_tr

        if mtx_type == "lockdown":
            cm_calc = CMCalculatorLockdown(data_tr=self.data_tr)
            self.get_sim_output = cm_calc.get_sim_output_cm_entries_lockdown
        elif mtx_type == "lockdown_3":
            cm_calc = CMCalculatorLockdownTypewise(data_tr=self.data_tr)
            self.get_sim_output = cm_calc.get_sim_output_cm_entries_lockdown_3

        super().__init__(sim_state, data_tr=data_tr)
        self.susc = sim_state["susc"]

        # Matrices of frequently used contact types
        self.contact_home = self.data_tr.contact_home
        self.contact_total = self.data_tr.contact_matrix

        self.upper_tri_size = int((self.data_tr.n_ag + 1) * self.data_tr.n_ag / 2)

        self.lhs_boundaries = cm_calc.lhs_boundaries

    def run(self):
        maxiter = 120000
        # check if r0_lhs contains < 1
        print("computing kappa for base_r0=" + str(self.base_r0))
        # get output from target calculator
        tar_out = TargetCalculator(data_tr=self.data_tr)
        r0_lhs_home = tar_out.get_output(cm_sim=self.contact_home)
        if r0_lhs_home[1] < 1:
            kappas = np.linspace(0, 1, 1000)
            r0_home_kappas = np.array(list(map(self.kappify, kappas)))
            k = np.argmax(r0_home_kappas > 1, axis=0)  # Returns the indices of the maximum values along an axis.
            kappa = kappas[k]
            print("kappa", kappa)

            # Get LHS table
            lhs_table = self._get_lhs_table(number_of_samples=120_000, kappa=kappa)

            results = list(tqdm(map(self.get_sim_output, lhs_table), total=lhs_table.shape[0]))
            results = np.array(results)

            # check if all r0s are > 1
            # res_min = results[:, 137].min()
            res_min = results[:, self.upper_tri_size - 1].min()
            if res_min < 1:
                print("minimal lhs_r0: " + str(res_min))

            # Sort tables by R0 values
            r0_col_idx = int(self.data_tr.upper_tri_size - 1)
            # r0_col_idx = int(self.upper_tri_size + 1)
            sorted_idx = results[:, r0_col_idx].argsort()
            results = results[sorted_idx]
            lhs_table = np.array(lhs_table[sorted_idx])
            sim_output = np.array(results)
            sleep(0.3)

            # Save outputs
            self._save_output(output=lhs_table, folder_name='lhs')  # (12, 000, 136)
            self._save_output(output=sim_output, folder_name='simulations')  # (12, 000, 136)

    def gen_icu_max(self, line) -> np.ndarray:
        time = np.arange(0, 1000, 0.5)
        cm = get_rectangular_matrix_from_upper_triu(line, self.data_tr.n_ag)
        solution = self.data_tr.model.get_solution(t=time, parameters=self.data_tr.params, cm=cm)
        icu_max = solution[:, self.data_tr.model.c_idx["ic"] *
                              self.data_tr.n_ag:(self.data_tr.model.c_idx["ic"] + 1) * self.data_tr.n_ag].max()
        return icu_max

    def kappify(self, kappa=None) -> float:
        cm_diff = self.data_tr.contact_matrix - self.data_tr.contact_home
        cm_sim = self.data_tr.contact_home + kappa * cm_diff

        # get output from target calculator
        tar_out = TargetCalculator(data_tr=self.data_tr)
        r0_lhs_home_k = tar_out.get_output(cm_sim=cm_sim)
        return r0_lhs_home_k[1]

    def _get_variable_parameters(self):
        return [str(self.susc), str(self.base_r0), format(self.beta, '.5f'), self.type]

    def _get_upper_bound_factor_unit(self) -> np.ndarray:
        cm_diff = (self.data_tr.contact_matrix - self.contact_home) * self.data_tr.age_vector
        min_diff = np.min(cm_diff) / 2
        return min_diff
