import os

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
    def __init__(self, data: DataLoader, n_samples: int = 1, country: str = "usa",
                 epi_model: str = "rost_model") -> None:
        self.country = country
        super().__init__(data=data, epi_model=epi_model, country=country)
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
                # SAMPLING
                if generate_lhs:
                    sampler_npi = src.SamplerNPI(
                        sim_obj=self,
                        target="epidemic_size", epi_model=self.epi_model,
                        country=self.country)
                    sampler_npi.run()

    def generate_analysis_results(self):
        # Update params by susceptibility vector
        susceptibility = np.ones(self.n_ag)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            self.params.update({"susc": susceptibility})
            # Update params by calculated BASELINE beta
            for base_r0 in self.r0_choices:
                self.prepare_simulations(base_r0=base_r0, susc=susc)
                analysis = ContactManipulation(sim_obj=self, susc=susc, base_r0=base_r0, model="rost")
                return analysis.run_plots()

    def calculate_prcc_values(self):
        # read files from the generated folder based on the given parameters
        sim_folder, lhs_folder = "simulations", "lhs"
        for root, dirs, files in os.walk("./sens_data/" + sim_folder):
            for filename in files:
                saved_simulation = np.loadtxt(
                    "./sens_data/" + sim_folder + "/" +
                    filename,
                    delimiter=';')
                saved_lhs_values = np.loadtxt(
                    "./sens_data/" + lhs_folder + "/" +
                    filename.replace("simulations", "lhs"),
                    delimiter=';')

                susc = float(filename.split("_")[2])
                base_r0 = float(filename.split("_")[3])

                # CALCULATIONS
                # Calculate PRCC values
                prcc_calculator = src.prcc_calculator.PRCCCalculator(sim_obj=self)
                prcc_calculator.calculate_prcc_values(
                    lhs_table=saved_lhs_values,
                    sim_output=saved_simulation)

                # calculate p-values
                prcc_calculator.calculate_p_values()
                stack_prcc_pval = np.hstack(
                    [prcc_calculator.prcc_list, prcc_calculator.p_value]
                ).reshape(-1, self.upper_tri_size).T

                # aggregate PRCC values
                prcc_calculator.aggregate_prcc_values()
                stack_value = np.hstack(
                    [prcc_calculator.agg_prcc, prcc_calculator.agg_std]
                ).reshape(-1, self.n_ag).T
                # CALCULATIONS END

                # save PRCC values
                os.makedirs("./sens_data/PRCC_Pvalues", exist_ok=True)
                fname = "_".join([str(susc), str(base_r0)])
                filename = "sens_data/PRCC_Pvalues" + "/" + fname
                np.savetxt(fname=filename + ".csv", X=stack_prcc_pval, delimiter=";")

                # save PRCC p-values
                os.makedirs("./sens_data/agg_prcc", exist_ok=True)
                filename = "sens_data/agg_prcc" + "/" + "_".join([fname])
                np.savetxt(fname=filename + ".csv", X=stack_value, delimiter=";")

    def plot_prcc_values(self):
        for susc in self.susc_choices:
            for base_r0 in self.r0_choices:
                print(susc, base_r0)
                # read files from the generated folder based on the given parameters
                agg_values = ["PRCC_Pvalues", "agg_prcc"]  # for plotting the aggregation methods
                for agg in agg_values:
                    for root, dirs, files in os.walk("./sens_data/" + agg):
                        for filename in files:
                            filename_without_ext = os.path.splitext(filename)[0]
                            # load prcc-pvalues
                            saved_prcc_pval = np.loadtxt(
                                "./sens_data/" + agg + "/" + filename,
                                delimiter=';')

                            # Plot results
                            plotter = src.Plotter(sim_obj=self, data=self.data)
                            # plot contact matrices for different models
                            plotter.plot_contact_matrices_models(
                                filename="contact",
                                model="rost",
                                contact_data=self.data.contact_data,
                                n_ag=self.n_ag)

                            if agg == "PRCC_Pvalues":
                                # plot prcc values
                                plotter.generate_prcc_p_values_heatmaps(
                                    prcc_vector=saved_prcc_pval[:, 0],
                                    p_values=saved_prcc_pval[:, 1],
                                    filename_without_ext=filename_without_ext)
                            else:
                                # plot aggregation values
                                plotter.plot_aggregation_prcc_pvalues(
                                    prcc_vector=saved_prcc_pval[:, 0],
                                    std_values=saved_prcc_pval[:, 1],
                                    filename_without_ext=filename_without_ext)

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
                    f"./sens_data/Epidemic_size/Epidemic_values",
                    f"./sens_data/Epidemic_peak/Peak_values",
                    f"./sens_data/icu/icu_values",
                    f"./sens_data/death_size/death_values",
                    f"./sens_data/hospital/hosp_values"
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
                plot.plot_model_max_values(max_values=dfs, model="rost")

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
