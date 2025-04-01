from functools import partial
import numpy as np
from tqdm import tqdm

import src
from src.sampling.cm_calculator_lockdown import CMCalculatorLockdown
from src.sampling.target.ODE_target_calculator import ODETargetCalculator
from src.sampling.sampler_base import SamplerBase
from src.sampling.target.r0_target_calculator import R0TargetCalculator


class SamplerNPI(SamplerBase):
    def __init__(self, sim_obj: src.SimulationNPI, country: str,
                 epi_model: str, config) -> None:
        super().__init__(sim_obj=sim_obj, config=config)
        self.country = country
        self.epi_model = epi_model
        self.sim_obj = sim_obj

        cm_calc = CMCalculatorLockdown(sim_obj=self.sim_obj)
        self.get_sim_output = cm_calc.get_sim_output_cm_entries_lockdown
        self.susc = sim_obj.sim_state["susc"]
        self.lhs_boundaries = cm_calc.lhs_boundaries

        self.calc = None

    def run(self):
        number_of_samples = self.sim_obj.n_samples
        lhs_table = self._get_lhs_table(number_of_samples=number_of_samples)

        print(f"Simulation for {self.epi_model} model, "
              f"contact_matrix: {self.country}, "
              f"sample_size: {number_of_samples}, "
              f"susc: {self.susc}, "
              f"base_r0: {self.base_r0}.")

        self.calc = ODETargetCalculator(sim_obj=self.sim_obj,
                                        epi_model=self.sim_obj.epi_model,
                                        config=self.config)

        # Run the simulations using the LHS table and the contact matrix builder
        results = list(tqdm(map(partial(self.get_sim_output, calc=self.calc),
                                lhs_table),
                            total=lhs_table.shape[0]))
        sim_outputs = np.array(results)

        # Optionally compute R0 values and append to simulation output
        if self.config["include_r0"]:
            self.calc = R0TargetCalculator(sim_obj=self.sim_obj,
                                           country=self.country)
            r0_results = list(tqdm(map(partial(self.get_sim_output, calc=self.calc),
                                       lhs_table),
                                   total=lhs_table.shape[0]))
            r0_array = np.array(r0_results)[:, self.sim_obj.upper_tri_size:].reshape(-1, 1)
            sim_outputs = np.append(sim_outputs, r0_array, axis=1)

        # Save results
        self._save_output(output=lhs_table, folder_name="lhs")
        self._save_output(output=sim_outputs, folder_name="simulations")
        self._save_output_json(folder_name="simulations")

    def _get_variable_parameters(self):
        return [str(self.susc), str(self.base_r0), format(self.beta, '.5f')]