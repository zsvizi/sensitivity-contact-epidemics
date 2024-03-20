import os
import json
import numpy as np
import pandas as pd

import src
from src.contact_manipulation import ContactManipulation
from src.dataloader import DataLoader
from src.chikina.r0 import R0SirModel
from src.model.r0_generator import R0Generator
from src.moghadas.r0 import R0SeyedModel
from src.seir.r0 import R0SeirSVModel
from src.simulation_base import SimulationBase


class SimulationNPI(SimulationBase):
    def __init__(self, data: DataLoader, n_samples: int = 1,
                 country: str = "usa",
                 epi_model: str = "rost_model") -> None:

        if epi_model in ["rost", "chikina", "moghadas"]:
            self.config = {
                "include_final_death_size": True,
                "include_icu_peak": True,
                "include_hospital_peak": True,
                "include_infecteds_peak": True,
                "include_infecteds": False,  # excluded from the targets
                "include_r0": True
            }
        elif epi_model == "seir":
            self.config = {
                "include_final_death_size": False,
                "include_icu_peak": False,
                "include_hospital_peak": False,
                "include_infecteds_peak": True,
                "include_infecteds": False,  # excluded from the targets
                "include_r0": True
            }
        else:
            raise ValueError("Invalid epi_model")

        self.country = country
        super().__init__(data=data, epi_model=epi_model,
                         country=country)
        self.n_samples = n_samples
        self.epi_model = epi_model

        # User-defined parameters
        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.2, 1.8, 2.5]

    def generate_lhs(self, generate_lhs: bool = True):
        # Update params by susceptibility vector
        susceptibility = np.ones(self.n_ag)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            self.params.update({"susc": susceptibility})
            # Update params by calculated BASELINE beta
            for base_r0 in self.r0_choices:
                self.prepare_simulations(base_r0=base_r0, susc=susc)
                if generate_lhs:
                    sampler_npi = src.SamplerNPI(
                        sim_obj=self,
                        epi_model=self.epi_model,
                        country=self.country, config=self.config)
                    sampler_npi.run()

    def calculate_prcc_values(self):
        sim_folder = "simulations"
        lhs_folder = "lhs"
        prcc_dir = "PRCC_Pvalues"
        agg_dir = "agg_prcc"

        # Read files from the generated JSON folder based on the given parameters
        for root, dirs, files in os.walk(os.path.join("./sens_data", sim_folder)):
            for filename in files:
                # Load JSON data
                saved_json_data = self._load_output_json(sim_folder, filename)
                susc, base_r0 = float(filename.split("_")[2]), float(filename.split("_")[3])

                # Load LHS table data separately
                lhs_filename = filename.replace("simulations",
                                                "lhs").replace(".json", ".csv")
                saved_lhs_table = np.loadtxt(
                    os.path.join("./sens_data", lhs_folder, lhs_filename),
                    delimiter=';'
                )
                # load simulation tables
                sim_filename = filename.replace(".json", ".csv")
                saved_simulation = np.loadtxt(
                    os.path.join("./sens_data", sim_folder, sim_filename),
                    delimiter=';'
                )
                # CALCULATIONS
                for key, value in saved_json_data.items():
                    prcc_calculator = src.prcc_calculator.PRCCCalculator(sim_obj=self)
                    prcc_calculator.calculate_prcc_values(
                        lhs_table=saved_lhs_table,
                        sim_output=saved_simulation[:, value]
                    )
                    prcc_calculator.calculate_p_values()
                    stack_prcc_pval = np.hstack(
                        [prcc_calculator.prcc_list, prcc_calculator.p_value]
                    ).reshape(-1, self.upper_tri_size).T

                    prcc_calculator.aggregate_prcc_values()
                    stack_value = np.hstack(
                        [prcc_calculator.agg_prcc, prcc_calculator.agg_std]
                    ).reshape(-1, self.n_ag).T
                    # CALCULATIONS END

                    # Save PRCC values
                    fname = "_".join([str(susc), str(base_r0)])
                    prcc_folder = os.path.join("./sens_data", prcc_dir, fname, "_" +
                                               key)
                    os.makedirs(prcc_folder, exist_ok=True)
                    prcc_fname = os.path.join(prcc_folder, "prcc.csv")
                    np.savetxt(fname=prcc_fname, X=stack_prcc_pval, delimiter=";")

                    # Save agg_prcc values and the std deviations
                    agg_folder = os.path.join("./sens_data", agg_dir, fname, "_" +
                                              key)
                    os.makedirs(agg_folder, exist_ok=True)
                    agg_fname = os.path.join(agg_folder, "agg.csv")
                    np.savetxt(fname=agg_fname, X=stack_value, delimiter=";")

    def _load_output_json(self, folder_name, filename):
        directory = os.path.join("./sens_data", folder_name)
        filename = os.path.join(directory, filename.replace(".csv", ".json"))
        with open(filename, "r") as json_file:
            data = json.load(json_file)
        return data

    def plot_prcc_values(self):
        agg_values = ["agg_prcc", "PRCC_Pvalues"]
        for susc in self.susc_choices:
            for base_r0 in self.r0_choices:
                print(susc, base_r0)
                for agg in agg_values:
                    agg_dir = f"./sens_data/{agg}/{susc}_{base_r0}"
                    susc_base_r0_parts = agg_dir.split("_")
                    base_r0_value = susc_base_r0_parts[-1]
                    for root, dirs, files in os.walk(agg_dir):
                        for filename in files:
                            plotter = src.Plotter(sim_obj=self, data=self.data)
                            # plot total contact matrices of the models
                            plotter.plot_contact_matrices_models(filename="contact",
                                                                 model=self.epi_model,
                                                                 contact_data=self.data.contact_data,
                                                                 plot_total_contact=True)

                            if filename == "prcc.csv":
                                saved_prcc_pval = np.loadtxt(os.path.join(root, filename), delimiter=';')
                                plotter.generate_prcc_p_values_heatmaps(
                                    prcc_vector=abs(saved_prcc_pval[:, 0]),
                                    p_values=abs(saved_prcc_pval[:, 1]),
                                    filename_without_ext=base_r0_value,
                                    option=root
                                )

                            elif filename == "agg.csv":
                                saved_prcc_pval = np.loadtxt(os.path.join(root, filename), delimiter=';')
                                plotter.plot_aggregation_prcc_pvalues(
                                    prcc_vector=saved_prcc_pval[:, 0],
                                    std_values=saved_prcc_pval[:, 1],
                                    filename_without_ext=base_r0_value,
                                    option=root
                                )

    def generate_analysis_results(self):
        # Update params by susceptibility vector
        susceptibility = np.ones(self.n_ag)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            self.params.update({"susc": susceptibility})
            # Update params by calculated BASELINE beta
            for base_r0 in self.r0_choices:
                self.prepare_simulations(base_r0=base_r0, susc=susc)
                analysis = ContactManipulation(sim_obj=self, susc=susc,
                                               base_r0=base_r0, model="rost")
                analysis.run_plots()

    def plot_max_values_contact_manipulation(self):
        """
        Loads and plots the maximum values for contact manipulation scenarios saved from running
        get analysis results method above.
        This method iterates over susceptibility (susc) and base R0 (base_r0) choices
        and load the CSV files saved in different directories representing scenarios of
        contact manipulation.
        """
        for susc in self.susc_choices:
            for base_r0 in self.r0_choices:
                print(susc, base_r0)
                # Define directories and aggregation values
                directories = [
                    f"./sens_data/Epidemic/Epidemic_values",
                    f"./sens_data/icu/icu_values",
                    f"./sens_data/death/death_values",
                    f"./sens_data/hospital/hospital_values"
                ]
                # Initialize dictionary to store DataFrames
                dfs = {}
                # Iterate over directories
                for directory in directories:
                    # Get the base directory name
                    base_dir = os.path.basename(directory)
                    # Load files from the directory
                    for filename in os.listdir(directory):
                        if filename.endswith(".csv"):  # Assuming CSV files
                            # Load data from file into NumPy array
                            saved_max_values = np.loadtxt(os.path.join(directory, filename), delimiter=';')

                            # Convert NumPy arrays into Pandas DataFrame
                            df = pd.DataFrame(saved_max_values)
                            # Set column name
                            column_name = os.path.splitext(filename)[0]
                            df.columns = [column_name]
                            # Add DataFrame to the dictionary
                            dfs[f"{base_dir}/{column_name}"] = df
                plot = src.Plotter(sim_obj=self, data=self.data)
                plot.plot_model_max_values(max_values=dfs, model="chikina")

    def prepare_simulations(self, base_r0, susc):
        r0generator = self.choose_r0_generator()
        r0 = r0generator.get_eig_val(
            contact_mtx=self.contact_matrix,
            susceptibles=self.susceptibles.reshape(1, -1),
            population=self.population
        )[0]
        beta = base_r0 / r0
        self.params.update({"beta": beta})
        self.sim_state.update(
            {"base_r0": base_r0,
             "beta": beta,
             "susc": susc,
             "r0generator": r0generator})

    def choose_r0_generator(self):
        if self.epi_model == "chikina":
            r0generator = R0SirModel(param=self.params)
        elif self.epi_model == "rost":
            r0generator = R0Generator(param=self.params)
        elif self.epi_model == "seir":
            r0generator = R0SeirSVModel(param=self.params)
        elif self.epi_model == "moghadas":
            r0generator = R0SeyedModel(param=self.params)
        else:
            raise Exception("No model is given!")
        return r0generator
