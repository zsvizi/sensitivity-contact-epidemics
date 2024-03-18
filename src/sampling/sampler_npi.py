from functools import partial
from time import sleep
import os

import numpy as np
from tqdm import tqdm

import src
from src.sampling.cm_calculator_lockdown import CMCalculatorLockdown
from src.sampling.ODE_target_calculator import ODETargetCalculator
from src.sampling.sampler_base import SamplerBase
from src.sampling.r0_target_calculator import R0TargetCalculator


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
        kappa = self.calculate_kappa()
        print("computing kappa for base_r0=" + str(self.base_r0))
        number_of_samples = self.sim_obj.n_samples
        lhs_table = self._get_lhs_table(number_of_samples=number_of_samples,
                                        kappa=kappa)
        # Initialize sim_outputs_combined as a copy of lhs_table
        print(f"Simulation for {self.target}: {number_of_samples} "
              f"samples ({self.susc}-{self.base_r0})")
        self.calc = ODETargetCalculator(sim_obj=self.sim_obj,
                                        epi_model=self.sim_obj.epi_model,
                                        config=self.config)

        # Calculate simulation output for the current target
        results = list(tqdm(map(partial(self.get_sim_output, calc=self.calc),
                                lhs_table),
                            total=lhs_table.shape[0]))
        sim_outputs = np.array(results)

        # Define column information
        column_info = {
            "target_names": ["final_death_size", "icu_peak", "hospital_peak", "infecteds_peak"],  # column names
            "column_descriptions": ["Run simulations with final_death_size as target",
                                    "Run simulations with icu_peak as target",
                                    "Run simulations with hospital_peak as target",
                                    "Run simulations with infecteds_peak as target"]
        }
        # Extract individual targets from sim_outputs
        final_death_size = sim_outputs[:, self.sim_obj.upper_tri_size + 0:
                                          self.sim_obj.upper_tri_size + 1]
        icu_peak = sim_outputs[:, self.sim_obj.upper_tri_size + 1:
                                  self.sim_obj.upper_tri_size + 2]
        hospital_peak = sim_outputs[:, self.sim_obj.upper_tri_size + 2:
                                       self.sim_obj.upper_tri_size + 3]
        infecteds_peak = sim_outputs[:, self.sim_obj.upper_tri_size + 3:
                                        self.sim_obj.upper_tri_size + 4]

        # Initialize empty dictionary to store simulation outputs
        sim_output_values = {}
        # Define simulation output values for each target
        if self.config["include_final_death_size"]:
            sim_output_values["final_death_size"] = {
                "values": final_death_size.flatten().tolist(),
                "description": "Run simulations with final_death_size as target"
            }

        if self.config["include_icu_peak"]:
            sim_output_values["icu_peak"] = {
                "values": icu_peak.flatten().tolist(),
                "description": "Run simulations with icu_peak as target"
            }

        if self.config["include_hospital_peak"]:
            sim_output_values["hospital_peak"] = {
                "values": hospital_peak.flatten().tolist(),
                "description": "Run simulations with hospital_peak as target"
            }

        if self.config["include_infecteds_peak"]:
            sim_output_values["infecteds_peak"] = {
                "values": infecteds_peak.flatten().tolist(),
                "description": "Run simulations with infecteds_peak as target"
            }

        if self.config["include_r0"]:
            self.calc = R0TargetCalculator(sim_obj=self.sim_obj,
                                           country=self.country)
            results = list(tqdm(map(partial(self.get_sim_output, calc=self.calc),
                                    lhs_table),
                                total=lhs_table.shape[0]))
            sim_outputs = np.append(sim_outputs,
                                    np.array(results)[:, self.sim_obj.upper_tri_size:].reshape(-1, 1),
                                    axis=1)

            r0 = sim_outputs[:, -1].flatten().tolist()  # Extract and update r0 values
            # Append r0 values to sim_output_values dictionary
            sim_output_values["r0"] = {
                "values": r0,
                "description": "Run simulations with r0 as target"
            }

            # Update column information for R0
            column_info["target_names"].append("r0")
            column_info["column_descriptions"].append("Run simulations with r0 as target")

            # Save simulation outputs and LHS table
            output_info = {"sim_output_values": sim_output_values}

        self._save_output(lhs_table, folder_name="lhs")
        self._save_output(sim_outputs, folder_name="simulations")
        self._save_output_json(sim_output_values, folder_name="simulations",
                          output_info=output_info)

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
