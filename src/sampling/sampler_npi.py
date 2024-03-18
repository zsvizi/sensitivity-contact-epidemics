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
            self.calc = ODETargetCalculator(sim_obj=self.sim_obj,
                                            epi_model=sim_obj.epi_model,
                                            config=self.config)
        self.susc = sim_obj.sim_state["susc"]
        self.lhs_boundaries = cm_calc.lhs_boundaries

    def run_r0(self):
        # Load previously saved simulation outputs from the "epidemic_size" phase
        lhs_folder, sim_folder = "lhs", "simulations"

        # Get the list of files in sim_folder
        sim_files = os.listdir(os.path.join("./sens_data", sim_folder))

        # Sort the files to ensure consistent order
        sim_files.sort()

        # Process files in order
        for filename in sim_files:
            sim_outputs = np.loadtxt(
                os.path.join("./sens_data", sim_folder, filename),
                delimiter=';'
            )
            lhs_table = np.loadtxt(
                os.path.join("./sens_data", lhs_folder,
                             filename.replace("simulations", "lhs")),
                delimiter=';'
            )
            if self.target == "r0":
                results = list(tqdm(map(partial(self.get_sim_output, calc=self.calc),
                                        lhs_table),
                                    total=lhs_table.shape[0]))
                r0_result = np.array(results)[:, -1]  # Extract the last column from r0_result

                # Append the last column from r0_result to sim_outputs
                sim_output = np.hstack((sim_outputs, r0_result.reshape(-1, 1)))
                # Save the combined results for 5 targets, 4 from epidemic size & r0
                np.savetxt(os.path.join("./sens_data", sim_folder, filename),
                           sim_output[:, :141], delimiter=';')

    def run(self):
        kappa = self.calculate_kappa()
        print("computing kappa for base_r0=" + str(self.base_r0))
        number_of_samples = self.sim_obj.n_samples
        lhs_table = self._get_lhs_table(number_of_samples=number_of_samples,
                                        kappa=kappa)
        # Initialize sim_outputs_combined as a copy of lhs_table
        sim_outputs = lhs_table.copy()
        # Calculate simulation output for the current target
        results = list(tqdm(map(partial(self.get_sim_output, calc=self.calc),
                                lhs_table),
                            total=lhs_table.shape[0]))
        if self.target == "epidemic_size":
            print(f"Simulation for {self.target}: {number_of_samples} "
                  f"samples ({self.susc}-{self.base_r0})")
            target_result = np.array(results)[:, self.sim_obj.upper_tri_size:]
            # Extracting all values from each dictionary in the list
            numpy_arrays = [[[val for val in d.values()] for d in sublist
                             if isinstance(d, dict)] for
                            sublist in target_result]

            # Accessing each array in the sublist
            sublist = [arr for sublist in numpy_arrays for arr in sublist[0]]

            # Reshape sublist into target columns
            sublist_uniform = [np.squeeze(arr) for arr in sublist]
            sublist_reshaped = np.array(sublist_uniform).reshape(number_of_samples, -1)
            sim_outputs = np.hstack((sim_outputs, sublist_reshaped))

            self._save_output(output=sim_outputs, folder_name="simulations")
            self._save_output(output=lhs_table, folder_name="lhs")
        return lhs_table, sim_outputs

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
