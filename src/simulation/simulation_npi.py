import os
import json
import numpy as np
import pandas as pd

import src
from src.dataloader import DataLoader
from src.examples import chikina, rost, moghadas, seir, validation
from src.simulation.contact_manipulation import ContactManipulation
from src.simulation.simulation_base import SimulationBase


class SimulationNPI(SimulationBase):
    """
    Main simulation class for Non-Pharmaceutical Intervention (NPI) sensitivity and scenario analysis.

    Handles model selection, LHS sampling, PRCC analysis, and visualization of epidemiological results.
    """

    def __init__(self, data: DataLoader, n_samples: int = 1, country: str = "usa", epi_model: str = "rost_prem",
                 strategy: str = "absolute", is_kappa_applied: bool = True,
                 contact_dispersion: float = 0.5, n: int = 188) -> None:
        """
        Initializes the simulation environment and epidemiological model.

        :param DataLoader data: Data loader containing demographic and contact data.
        :param int n_samples: Number of samples for the Latin Hypercube Sampling (LHS).
        :param str country: Country name used for parameter configuration.
        :param str epi_model: Epidemiological model identifier.
        :param str strategy: Contact manipulation strategy
        :param bool is_kappa_applied: Whether the kappa scaling factor is applied to contact matrices or not
        :param float contact_dispersion: The dispersion parameter used when sampling contact matrix values
        :param int n: The assumed survey sample size (number of participants) used to calculate the variance
                      during contact matrix sampling.
        """
        super().__init__(data=data, country=country)

        self.strategy = strategy
        self.is_kappa_applied = is_kappa_applied
        self.country = country
        self.n_samples = n_samples
        self.epi_model = epi_model
        self.contact_dispersion = contact_dispersion
        self.n = n

        self.config = None

        # Set up configuration for each model type
        self._set_up_config(epi_model=epi_model)

        # Define number of age groups with distinct susceptibilities per model
        self.model_susceptibility_ages = {
            "rost_prem": 4,
            "rost_maszk": 2,
            "seir": 4,
            "chikina": 4,
            "moghadas": 1,
            "validation": 1
        }

        # Initialize model
        self._choose_model(epi_model=epi_model)

        # Extract initial susceptible vector
        self.susceptibles = self.model.get_initial_values()[
            self.model.c_idx["s"] * self.n_ag: (self.model.c_idx["s"] + 1) * self.n_ag
        ]

        # User-defined control parameters
        self.susc_choices = [1.0]
        self.r0_choices = [2.5]  # Example: could include multiple base R_0 values

    def generate_lhs(self, generate_lhs: bool = True) -> None:
        """
        Generate Latin Hypercube Sampling (LHS) parameter sets for simulations.

        :param bool generate_lhs: Whether to execute the sampler (True) or only prepare parameters.
        """
        for susc in self.susc_choices:
            susceptibility = self.get_model_susceptibility(susc)
            self.params.update({"susc": susceptibility})
            for base_r0 in self.r0_choices:
                self.prepare_simulations(base_r0=base_r0, susc=susc)
                if generate_lhs:
                    sampler_npi = src.SamplerNPI(sim_obj=self)
                    sampler_npi.run()

    def calculate_prcc_values(self) -> None:
        """
        Compute and save Partial Rank Correlation Coefficients (PRCC) and their p-values
        from the LHS-generated simulations.
        """
        sim_folder = "simulations"
        lhs_folder = "lhs"
        prcc_dir = "PRCC_Pvalues"
        agg_dir = "agg_prcc"

        for root, dirs, files in os.walk(os.path.join("./sens_data", sim_folder)):
            for filename in files:
                saved_json_data = self._load_output_json(sim_folder, filename)
                susc, base_r0 = float(filename.split("_")[2]), float(filename.split("_")[3])

                # Load LHS and simulation results
                lhs_filename = filename.replace("simulations", "lhs").replace(".json", ".csv")
                saved_lhs_table = np.loadtxt(
                    os.path.join("./sens_data", lhs_folder, lhs_filename), delimiter=";"
                )
                sim_filename = filename.replace(".json", ".csv")
                saved_simulation = np.loadtxt(
                    os.path.join("./sens_data", sim_folder, sim_filename), delimiter=";"
                )

                for key, value in saved_json_data.items():
                    prcc_calculator = src.PRCCCalculator(sim_obj=self)
                    prcc_calculator.calculate_prcc_values(
                        lhs_table=saved_lhs_table,
                        sim_output=saved_simulation[:, value]
                    )
                    prcc_calculator.calculate_p_values()

                    stack_prcc_pval = np.hstack(
                        [prcc_calculator.prcc_list, prcc_calculator.p_value]
                    ).reshape(-1, self.upper_tri_size).T

                    prcc_calculator.aggregate_prcc_values_median()
                    stack_value = np.hstack(
                        [
                            prcc_calculator.agg_prcc,
                            prcc_calculator.confidence_lower,
                            prcc_calculator.confidence_upper,
                        ]
                    ).reshape(-1, self.n_ag).T

                    # Save PRCC and aggregated PRCC values
                    fname = f"{susc}_{base_r0}"
                    prcc_folder = os.path.join("./sens_data", prcc_dir, fname, "_" + key)
                    os.makedirs(prcc_folder, exist_ok=True)
                    np.savetxt(os.path.join(prcc_folder, "prcc.csv"), X=stack_prcc_pval, delimiter=";")

                    agg_folder = os.path.join("./sens_data", agg_dir, fname, "_" + key)
                    os.makedirs(agg_folder, exist_ok=True)
                    np.savetxt(os.path.join(agg_folder, "agg.csv"), X=stack_value, delimiter=";")

    def plot_prcc_values(self) -> None:
        """
        Visualize PRCC and aggregated PRCC results for all susceptibility and R_0 configurations.
        Generates heatmaps and aggregated correlation plots.
        """
        agg_values = ["agg_prcc", "PRCC_Pvalues"]
        for susc in self.susc_choices:
            for base_r0 in self.r0_choices:
                print(susc, base_r0)
                for agg in agg_values:
                    agg_dir = f"./sens_data/{agg}/{susc}_{base_r0}"
                    base_r0_value = agg_dir.split("_")[-1]
                    for root, dirs, files in os.walk(agg_dir):
                        for filename in files:
                            plotter = src.Plotter(sim_obj=self, data=self.data)

                            # Plot contact matrices
                            plotter.plot_contact_matrices_models(
                                filename="contact",
                                model=self.epi_model,
                                contact_data=self.data.contact_data,
                                plot_total_contact=False,
                            )
                            plotter.get_percentage_age_group_contact(
                                filename="mean_contact", model=self.epi_model
                            )

                            if filename == "prcc.csv":
                                saved_prcc_pval = np.loadtxt(os.path.join(root, filename), delimiter=";")
                                plotter.generate_prcc_p_values_heatmaps(
                                    prcc_vector=abs(saved_prcc_pval[:, 0]),
                                    p_values=abs(saved_prcc_pval[:, 1]),
                                    filename_without_ext=base_r0_value,
                                    option=root,
                                    model=self.epi_model,
                                )
                            elif filename == "agg.csv":
                                saved_prcc_pval = np.loadtxt(os.path.join(root, filename), delimiter=";")
                                plotter.plot_aggregation_prcc_pvalues(
                                    prcc_vector=saved_prcc_pval[:, 0],
                                    std_values=None,
                                    conf_lower=saved_prcc_pval[:, 1],
                                    conf_upper=saved_prcc_pval[:, 2],
                                    filename_without_ext=base_r0_value,
                                    model=self.epi_model,
                                    option=root,
                                )

    def generate_analysis_results(self) -> None:
        """
        Run contact manipulation analyses (varying contact intensities across groups)
        for each susceptibility and R_0 configuration, and generate the corresponding plots.
        """
        for susc in self.susc_choices:
            susceptibility = self.get_model_susceptibility(susc)
            self.params.update({"susc": susceptibility})
            for base_r0 in self.r0_choices:
                self.prepare_simulations(base_r0=base_r0, susc=susc)
                analysis = ContactManipulation(sim_obj=self, susc=susc, base_r0=base_r0)
                analysis.run_plots()

    def plot_max_values_contact_manipulation(self) -> None:
        """
        Load and plot maximum outcome values (e.g., ICU, final deaths size)
        from the contact manipulation analyses.
        """
        for susc in self.susc_choices:
            for base_r0 in self.r0_choices:
                print(susc, base_r0)

                if self.epi_model in ["rost_prem", "chikina", "moghadas"]:
                    directories = [
                        "./sens_data/Epidemic/Epidemic_values",
                        "./sens_data/icu/icu_values",
                        "./sens_data/death/death_values",
                        "./sens_data/hospital/hospital_values",
                    ]
                else:
                    directories = ["./sens_data/Epidemic/Epidemic_values"]

                dfs = {}
                for directory in directories:
                    base_dir = os.path.basename(directory)
                    for filename in os.listdir(directory):
                        if filename.endswith(".csv"):
                            saved_max_values = np.loadtxt(
                                os.path.join(directory, filename), delimiter=";"
                            )
                            df = pd.DataFrame(saved_max_values)
                            column_name = os.path.splitext(filename)[0]
                            df.columns = [column_name]
                            dfs[f"{base_dir}/{column_name}"] = df

                plot = src.Plotter(sim_obj=self, data=self.data)
                plot.plot_model_max_values(max_values=dfs, model=self.epi_model)

    def _set_up_config(self, epi_model: str) -> None:
        """
        Configure which model outputs to include in analyses based on the model type.

        :param str epi_model: Epidemiological model name.
        """
        if epi_model in ["rost_maszk", "rost_prem", "chikina", "moghadas"]:
            self.config = {
                "include_final_death_size": True,
                "include_icu_peak": True,
                "include_hospital_peak": True,
                "include_infecteds_peak": True,
                "include_infecteds": False,
                "include_r0": True
            }
        elif epi_model == "seir":
            self.config = {
                "include_final_death_size": False,
                "include_icu_peak": False,
                "include_hospital_peak": False,
                "include_infecteds_peak": True,
                "include_infecteds": False,
                "include_r0": True
            }
        elif epi_model == "validation":
            self.config = {
                "include_final_death_size": True,
                "include_icu_peak": False,
                "include_hospital_peak": False,
                "include_infecteds_peak": True,
                "include_infecteds": False,
                "include_r0": True
            }
        else:
            raise ValueError("Invalid epi_model")

    def _choose_model(self, epi_model: str) -> None:
        """
        Initialize the correct model object based on the selected epidemiological model.

        :param str epi_model: Model identifier string.
        """
        if epi_model in ["rost_maszk", "rost_prem"]:
            self.model = rost.RostModelHungary(model_data=self.data)
        elif epi_model == "chikina":
            self.model = chikina.SirModel(model_data=self.data)
        elif epi_model == "seir":
            self.model = seir.SeirUK(model_data=self.data)
        elif epi_model == "moghadas":
            self.model = moghadas.MoghadasModelUsa(model_data=self.data)
        elif epi_model == "validation":
            self.model = validation.ValidationModel(model_data=self.data)
        else:
            raise Exception("No model was given!")

    def get_model_susceptibility(self, susc_value: float) -> np.ndarray:
        """
        Construct a susceptibility vector appropriate for the current model type.

        :param float susc_value: Base susceptibility multiplier for the first few age groups.
        :return np.ndarray: Susceptibility array for all age groups.
        """
        n_susc_ages = self.model_susceptibility_ages[self.epi_model]
        susceptibility = np.ones(self.n_ag)
        susceptibility[:n_susc_ages] = susc_value
        return susceptibility

    def prepare_simulations(self, base_r0: float, susc: float) -> None:
        """
        Calibrate and update model parameters (e.g., beta) to achieve a target base R_0.

        :param float base_r0: Desired basic reproduction number.
        :param float susc: Susceptibility factor used in the calibration.
        """
        r0generator = self.choose_r0_generator()
        r0 = r0generator.get_eig_val(
            contact_mtx=self.contact_matrix,
            susceptibles=self.susceptibles.reshape(1, -1),
            population=self.population
        )[0]

        beta = base_r0 / r0
        self.params.update({"beta": beta})
        self.sim_state.update({
            "base_r0": base_r0,
            "beta": beta,
            "susc": susc,
            "r0generator": r0generator
        })

    @staticmethod
    def _load_output_json(folder_name: str, filename: str) -> dict:
        """
        Load a JSON output file generated during simulation or LHS sampling.

        :param str folder_name: Folder path within './sens_data'.
        :param str filename: Name of the target file.
        :return dict: JSON contents as a Python dictionary.
        """
        directory = os.path.join("./sens_data", folder_name)
        filename = os.path.join(directory, filename.replace(".csv", ".json"))
        with open(filename, "r") as json_file:
            data = json.load(json_file)
        return data

    def choose_r0_generator(self):
        """
        Select and return the R₀ generator corresponding to the active model type.

        :return object: Instance of an R₀ generator class.
        """
        if self.epi_model == "chikina":
            return chikina.R0SirModel(param=self.params)
        elif self.epi_model in ["rost_maszk", "rost_prem"]:
            return rost.R0Generator(param=self.params, n_age=self.n_ag)
        elif self.epi_model == "seir":
            return seir.R0SeirSVModel(param=self.params)
        elif self.epi_model == "moghadas":
            return moghadas.R0SeyedModel(param=self.params)
        elif self.epi_model == "validation":
            return validation.R0ValidationModel(param=self.params)
        else:
            raise Exception("No model is given!")
