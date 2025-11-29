from functools import partial
import numpy as np
from tqdm import tqdm

import src
from src.sampling.cm_calculator_lockdown import CMCalculatorLockdown
from src.sampling.target.ODE_target_calculator import ODETargetCalculator
from src.sampling.sampler_base import SamplerBase
from src.sampling.target.r0_target_calculator import R0TargetCalculator


class SamplerNPI(SamplerBase):
    """
    A sampler class for Non-Pharmaceutical Intervention (NPI) simulations.

    This class handles:
    - Generating Latin Hypercube Samples (LHS) of parameters and contact matrices
    - Running epidemic model simulations (ODE or R_0 based)
    - Managing kappa scaling for contact matrices (optional)
    - Saving the simulation results
    """

    def __init__(self, sim_obj: src.SimulationNPI) -> None:
        """
        Initializes the NPI sampler with the given simulation object.

        :param src.SimulationNPI sim_obj: Simulation object containing model setup,
                                          configuration, and state information.
        """
        super().__init__(sim_obj=sim_obj)
        self.sim_obj = sim_obj

        # Extract simulation configuration
        self.country = sim_obj.country
        self.epi_model = sim_obj.epi_model
        self.is_kappa_applied = sim_obj.is_kappa_applied
        self.strategy = sim_obj.strategy
        self.susc = sim_obj.sim_state["susc"]

        # Initialize lockdown-specific contact matrix calculator
        cm_calc = CMCalculatorLockdown(sim_obj=self.sim_obj)
        self.get_sim_output = cm_calc.get_sim_output_cm_entries_lockdown
        self.lhs_boundaries = cm_calc.lhs_boundaries

        # Placeholder for the selected target calculator (ODE or R_0)
        self.calc = None

    def run(self) -> None:
        """
        Executes the sampling and simulation workflow.

        Steps:
        1. Compute or skip the kappa scaling factor.
        2. Generate the Latin Hypercube Sample (LHS) based on the defined strategy.
        3. Run the selected epidemiological model for each sampled configuration.
        4. Optionally compute the basic reproduction number (R_0) for each run.
        5. Save all simulation and LHS outputs
        """
        # Compute or skip kappa scaling based on configuration
        kappa = self.get_kappa()

        # Generate Latin Hypercube Sampling table for contact matrix variation
        lhs_table = self._get_lhs_table(
            number_of_samples=self.sim_obj.n_samples,
            kappa=kappa,
            model=self.epi_model,
            strategy=self.strategy
        )

        print(f"Simulation for {self.epi_model} model, "
              f"contact_matrix: {self.country}, "
              f"sample_size: {self.sim_obj.n_samples}, "
              f"susc: {self.susc}, "
              f"base_r0: {self.base_r0}.")

        # Initialize the ODE-based target calculator (used for final death size, infected peak, etc.)
        self.calc = ODETargetCalculator(sim_obj=self.sim_obj)

        # Compute simulation outputs for all LHS samples
        # tqdm is used to display progress during (potentially) long simulations
        results = list(tqdm(
            map(partial(self.get_sim_output, calc=self.calc, strategy=self.strategy),
                lhs_table),
            total=lhs_table.shape[0]
        ))

        # Convert list of outputs into NumPy array
        sim_outputs = np.array(results)

        # Optionally compute and append R_0 outputs for each sample
        if self.config["include_r0"]:
            self.calc = R0TargetCalculator(sim_obj=self.sim_obj)
            results = list(tqdm(
                map(partial(self.get_sim_output, calc=self.calc, strategy=self.strategy),
                    lhs_table),
                total=lhs_table.shape[0]
            ))
            # Append only the R_0-related outputs to the simulation array
            sim_outputs = np.append(
                sim_outputs,
                np.array(results)[:, self.sim_obj.upper_tri_size:],
                axis=1
            )

        # Save LHS table, simulation results, and metadata
        self._save_output(output=lhs_table, folder_name="lhs")
        self._save_output(output=sim_outputs, folder_name="simulations")
        self._save_output_json(folder_name="simulations")

    def get_kappa(self) -> float:
        """
        Determines whether to calculate kappa (scaling factor for non-home contacts)
        based on simulation configuration.

        :return float: The calculated or skipped (0.0) kappa value.
        """
        if self.is_kappa_applied:
            kappa = self.calculate_kappa()
            print(f"Computing kappa for base_r0 = {self.base_r0}")
        else:
            kappa = 0
            print("Skipping kappa calculation.")
        return kappa

    def calculate_kappa(self) -> float:
        """
        Computes the scaling factor (kappa) for contact matrices that ensures
        the basic reproduction number (R_0) is close to 1 under baseline conditions.

        The function iteratively tests kappa values in [0, 1] and selects the
        smallest one that causes R₀ to exceed 1.

        :return float: Kappa value corresponding to R_0 ≈ 1.
        """
        kappas = np.linspace(0, 1, 1000)
        r0_home_kappas = np.array(list(map(self.kappify, kappas)))

        # Find the index where R_0 first surpasses 1
        k = np.argmax(r0_home_kappas > 1, axis=0)
        kappa = kappas[k]
        print("k =", kappa)
        return kappa

    def kappify(self, kappa: float = None) -> float:
        """
        Calculates the basic reproduction number (R_0) for a given scaling factor kappa.

        The scaling modifies the total contact matrix between home and non-home
        contacts using:
            CM_sim = CM_home + kappa * (CM_total - CM_home)

        :param float kappa: Scaling factor applied to non-home contact components.
        :return float: Computed R_0 for the adjusted contact matrix.
        """
        cm_diff = self.sim_obj.contact_matrix - self.sim_obj.contact_home
        cm_sim = self.sim_obj.contact_home + kappa * cm_diff

        tar_out_r0 = R0TargetCalculator(sim_obj=self.sim_obj)
        r0_lhs_home_k = tar_out_r0.get_output(cm=cm_sim)
        return r0_lhs_home_k

    def _get_variable_parameters(self) -> list:
        """
        Retrieve a list of key simulation parameters used for identification
        and metadata storage.

        :return list: List of string representations of the variable parameters.
        """
        return [str(self.susc), str(self.base_r0), format(self.beta, '.5f')]
