from functools import partial

import numpy as np
from tqdm import tqdm

import src
from src.sampling.cm_calculator_lockdown import CMCalculatorLockdown
from src.sampling.target.ODE_target_calculator import ODETargetCalculator
from src.sampling.sampler_base import SamplerBase
from src.sampling.target.r0_target_calculator import R0TargetCalculator


class SamplerNPI(SamplerBase):
    def __init__(self, sim_obj: src.SimulationNPI) -> None:
        super().__init__(sim_obj=sim_obj)

        self.sim_obj = sim_obj

        self.country = sim_obj.country
        self.epi_model = sim_obj.epi_model
        self.is_kappa_applied = sim_obj.is_kappa_applied
        self.strategy = sim_obj.strategy
        self.susc = sim_obj.sim_state["susc"]
        cm_calc = CMCalculatorLockdown(sim_obj=self.sim_obj)
        self.get_sim_output = cm_calc.get_sim_output_cm_entries_lockdown
        self.lhs_boundaries = cm_calc.lhs_boundaries

        self.calc = None

    def run(self):
        kappa = self.get_kappa()
        lhs_table = self._get_lhs_table(
            number_of_samples=self.sim_obj.n_samples,
            kappa=kappa,
            model=self.epi_model,
            strategy=self.strategy
        )

        # Initialize sim_outputs_combined as a copy of lhs_table
        print(f"Simulation for {self.epi_model} model, "
              f"contact_matrix: {self.country}, "
              f"sample_size: {self.sim_obj.n_samples}, "
              f"susc: {self.susc}, "
              f"base_r0: {self.base_r0}.")

        self.calc = ODETargetCalculator(sim_obj=self.sim_obj)

        # Calculate simulation output for the current target
        results = list(tqdm(map(partial(self.get_sim_output, calc=self.calc,
                                        strategy=self.strategy),
                                lhs_table),
                            total=lhs_table.shape[0]))
        sim_outputs = np.array(results)

        if self.config["include_r0"]:
            self.calc = R0TargetCalculator(sim_obj=self.sim_obj)
            results = list(tqdm(map(partial(self.get_sim_output, calc=self.calc,
                                            strategy=self.strategy),
                                    lhs_table),
                                total=lhs_table.shape[0]))
            sim_outputs = np.append(sim_outputs,
                                    np.array(results)[:,
                                    self.sim_obj.upper_tri_size:].reshape(-1, 1),
                                    axis=1)

        self._save_output(output=lhs_table, folder_name="lhs")
        self._save_output(output=sim_outputs, folder_name="simulations")
        self._save_output_json(folder_name="simulations")

    def get_kappa(self):
        if self.is_kappa_applied:
            kappa = self.calculate_kappa()
            print("computing kappa for base_r0=" + str(self.base_r0))
        else:
            kappa = None
            print("Skipping kappa calculation.")
        return kappa

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

        tar_out_r0 = R0TargetCalculator(sim_obj=self.sim_obj)
        r0_lhs_home_k = tar_out_r0.get_output(cm=cm_sim)
        return r0_lhs_home_k

    def _get_variable_parameters(self):
        return [str(self.susc), str(self.base_r0), format(self.beta, '.5f')]
