from functools import partial
from time import sleep

import numpy as np
from tqdm import tqdm

import src
from src.sampling.cm_calculator_lockdown import CMCalculatorLockdown
from src.sampling.final_size_target_calculator import FinalSizeTargetCalculator
from src.sampling.r0_target_calculator import R0TargetCalculator
from src.sampling.sampler_base import SamplerBase


class SamplerNPI(SamplerBase):
    def __init__(self, sim_state: dict, sim_obj: src.SimulationNPI,
                 target: str = "r0", n_samples: int = 120000) -> None:
        super().__init__(sim_state=sim_state, sim_obj=sim_obj)
        self.target = target
        self.n_samples = n_samples

        self.__prepare_sampling()

    def __prepare_sampling(self):
        self.susc = self.sim_state["susc"]
        cm_calc = CMCalculatorLockdown(sim_obj=self.sim_obj, sim_state=self.sim_state)
        self.get_sim_output = cm_calc.get_sim_output_cm_entries_lockdown
        self.lhs_boundaries = cm_calc.lhs_boundaries

        self.r0_calculator = R0TargetCalculator(
            sim_obj=self.sim_obj,
            sim_state=self.sim_state)

        if self.target == "r0":
            self.calc = self.r0_calculator
            self.r0_lhs_home = self.calc.get_output(cm=self.sim_obj.contact_home)
        elif self.target == "epidemic_size":
            self.calc = FinalSizeTargetCalculator(sim_obj=self.sim_obj)
            self.final_size = self.calc.get_output(cm=self.sim_obj.contact_matrix)

        # Matrices of frequently used contact types
        self.contact_home = self.sim_obj.contact_home
        self.contact_total = self.sim_obj.contact_matrix

        self.upper_tri_size = int((self.sim_obj.n_ag + 1) * self.sim_obj.n_ag / 2)

    def run(self):
        kappa = self.calculate_kappa()
        # check if r0_lhs contains < 1
        print("computing kappa for base_r0=" + str(self.base_r0))
        lhs_table = self._get_lhs_table(number_of_samples=self.n_samples, kappa=kappa)

        # Results have shape of (number_of_samples, 136 + 1 + 1 + 16)
        results = np.array(list(tqdm(
            map(partial(self.get_sim_output, calc=self.calc),
                lhs_table),
            total=lhs_table.shape[0])
        ))

        # Sort tables by R0 values
        r0_col_idx = int(self.upper_tri_size - 1 + 2)  # r0 position
        sorted_idx = results[:, r0_col_idx].argsort()
        results = results[sorted_idx]
        lhs_table = np.array(lhs_table[sorted_idx])
        sleep(0.3)

        # Save outputs
        self._save_output(output=lhs_table, folder_name='lhs')
        self._save_output(output=results, folder_name='simulations')
        if self.target == "epidemic_size":
            self.calc: FinalSizeTargetCalculator
            self._save_output(output=self.calc.age_deaths, folder_name='age_deaths')

    def calculate_kappa(self):
        kappas = np.linspace(0, 1, 1000)
        r0_home_kappas = np.array(list(
            map(self.calculate_r0_for_optimal_kappa, kappas)
        ))
        # Returns the indices of the maximum values along an axis.
        k = np.argmax(r0_home_kappas > 1, axis=0)
        return kappas[k]

    def calculate_r0_for_optimal_kappa(self, kappa: float = None) -> float:
        cm_diff = self.sim_obj.contact_matrix - self.sim_obj.contact_home
        cm_sim = self.sim_obj.contact_home + kappa * cm_diff
        r0_lhs_home_k = self.r0_calculator.get_output(cm=cm_sim)
        return r0_lhs_home_k[1]

    def _get_variable_parameters(self):
        return [str(self.susc), str(self.base_r0), format(self.beta, '.5f')]
