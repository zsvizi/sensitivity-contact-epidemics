from functools import partial
from time import sleep

import numpy as np
from tqdm import tqdm

import src
from src.sampling.cm_calculator_lockdown import CMCalculatorLockdown
from src.sampling.final_size_target_calculator import FinalSizeTargetCalculator
from src.sampling.sampler_base import SamplerBase
from src.sampling.r0_target_calculator import R0TargetCalculator


class SamplerNPI(SamplerBase):
    def __init__(self, sim_obj: src.SimulationNPI, country: str,
                 epi_model: str, config, target: str = "r0") -> None:
        super().__init__(sim_obj=sim_obj, config=config)
        self.country = country
        self.epi_model = epi_model
        self.sim_obj = sim_obj
        self.target = target

        cm_calc = CMCalculatorLockdown(sim_obj=self.sim_obj)
        self.get_sim_output = cm_calc.get_sim_output_cm_entries_lockdown

        if self.target == "r0":
            self.calc = R0TargetCalculator(sim_obj=self.sim_obj,
                                           country=self.country)
        elif self.target == "epidemic_size":
            self.calc = FinalSizeTargetCalculator(sim_obj=self.sim_obj,
                                                  epi_model=sim_obj.epi_model,
                                                  config=self.config)
        self.susc = sim_obj.sim_state["susc"]
        self.lhs_boundaries = cm_calc.lhs_boundaries

    def run(self, option):
        kappa = self.calculate_kappa()
        print("computing kappa for base_r0=" + str(self.base_r0))
        number_of_samples = self.sim_obj.n_samples
        lhs_table = self._get_lhs_table(number_of_samples=number_of_samples,
                                        kappa=kappa)
        sim_output = None
        if (self.target == "epidemic_size" and (option in self.config and self.config[option])) or \
           (not self.target == "epidemic_size"):
            results = list(tqdm(
                map(partial(self.get_sim_output, calc=self.calc),
                    lhs_table),
                total=lhs_table.shape[0]))
            results = np.array(results)

            r0_col_idx = int(self.sim_obj.upper_tri_size)
            res_min = results[:, r0_col_idx].min()
            if res_min < 1:
                print("minimal lhs_r0: " + str(res_min))
            sorted_idx = results[:, r0_col_idx].argsort()
            results = results[sorted_idx]
            lhs_table = np.array(lhs_table[sorted_idx])
            sim_output = np.array(results)
            sleep(0.3)
            self._save_output(output=lhs_table, option=option, folder_name="lhs")
            self._save_output(output=sim_output, option=option, folder_name="simulations")
        return lhs_table, sim_output

    def calculate_kappa(self):
        kappas = np.linspace(0, 1, 1000)
        r0_home_kappas = np.array(list(map(self.kappify, kappas)))
        k = np.argmax(r0_home_kappas > 1, axis=0)  # Returns the indices of
        # the maximum values along an axis.
        kappa = kappas[k]
        print("k", kappa)
        return kappa

    def kappify(self, kappa: float = None) -> float:
        cm_diff = self.sim_obj.contact_matrix - self.sim_obj.contact_home
        cm_sim = self.sim_obj.contact_home + kappa * cm_diff

        tar_out_r0 = R0TargetCalculator(sim_obj=self.sim_obj, country=self.country)
        r0_lhs_home_k = tar_out_r0.get_output(cm=cm_sim)
        return r0_lhs_home_k

    def _get_variable_parameters(self):
        return [str(self.susc), str(self.base_r0), format(self.beta, '.5f')]
