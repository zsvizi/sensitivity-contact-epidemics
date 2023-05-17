import os
import itertools

import numpy as np

import src
from src.dataloader import DataLoader
from src.model.r0_generator import R0Generator
from src.simulation_base import SimulationBase


class SimulationNPI(SimulationBase):
    def __init__(self, data: DataLoader) -> None:
        super().__init__(data=data)

        # User-defined parameters
        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.2, 1.8, 2.5]

        self.lhs_table = None
        self.sim_output = None
        self.prcc_values = None

        self.n_samples = 120000

    def generate_lhs(self):
        # 1. Update params by susceptibility vector
        susceptibility = np.ones(16)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            self.params.update({"susc": self.susceptibles})
            self.sim_state.update({"susc": susceptibility})
            # 2. Update params by calculated BASELINE beta
            for base_r0 in self.r0_choices:
                r0generator = R0Generator(param=self.params)
                beta = base_r0 / r0generator.get_eig_val(
                    contact_mtx=self.contact_matrix,
                    susceptibles=self.susceptibles.reshape(1, -1),
                    population=self.population
                )[0]
                self.params.update({"beta": beta})
                self.sim_state.update(
                    {"base_r0": base_r0,
                     "beta": beta,
                     "susc": susc,
                     "r0generator": r0generator})

                # Execute sampling for independent parameters
                sampler_npi = src.SamplerNPI(
                    sim_state=self.sim_state,
                    sim_obj=self,
                    n_samples=self.n_samples,
                    target="epidemic_size")
                self.lhs_table, self.sim_output = sampler_npi.run()

    def calculate_prcc_values(self):
        # read files from the generated folder based on the given parameters
        sim_folder, lhs_folder = "simulations", "lhs"
        for root, dirs, files in os.walk("./sens_data/" + sim_folder):
            for filename in files:
                saved_simulation = np.loadtxt("./sens_data/" + sim_folder + "/" +
                                              filename, delimiter=';')
                saved_lhs_values = np.loadtxt("./sens_data/" + lhs_folder + "/" +
                                              filename.replace("simulations", "lhs"), delimiter=';')

                susc = float(filename.split("_")[2])
                base_r0 = float(filename.split("_")[3])

                # Calculate PRCC values
                prcc_calculator = src.prcc_calculator.PRCCCalculator(
                    number_of_samples=self.n_samples,
                    sim_obj=self)
                prcc_calculator.calculate_prcc_values(
                    lhs_table=saved_lhs_values,
                    sim_output=saved_simulation)
                # calculate p-values
                prcc_calculator.calculate_p_values()

                stack_prcc_pval = np.hstack([prcc_calculator.prcc_list, prcc_calculator.p_value]).reshape(2, 136).T
                os.makedirs("./sens_data/PRCC_Pvalues", exist_ok=True)
                fname = "_".join([str(susc), str(base_r0)])
                filename = "sens_data/PRCC_Pvalues" + "/" + fname
                np.savetxt(fname=filename + ".csv", X=stack_prcc_pval, delimiter=";")

                # aggregate PRCC values
                self.aggregate_prcc_values(prcc_calculator=prcc_calculator,
                                           fname=fname)
                options = ["first", "second", "third"]
                for option in options:
                    agg_option = prcc_calculator.new_prcc_pval_aggregation(option=option)
                    stack_value = np.hstack([prcc_calculator.agg_option, prcc_calculator.agg_var]).reshape(2, 16).T
                    os.makedirs("./sens_data/agg_values_options", exist_ok=True)
                    filename = "sens_data/agg_values_options" + "/" + "_".join([fname, option])
                    np.savetxt(fname=filename + ".csv", X=stack_value, delimiter=";")

    def plot_prcc_values(self):
        for susc in self.susc_choices:
            for base_r0 in self.r0_choices:
                if self.prcc_values is None:
                    print(susc, base_r0)
                    # read files from the generated folder based on the given parameters
                    prcc_pvalues, agg_values = "PRCC_Pvalues", "agg_values_options"
                    for root, dirs, files in os.walk("./sens_data/" + "PRCC_Pvalues"):
                        for filename in files:
                            filename_without_ext = os.path.splitext(filename)[0]
                            # load prcc-pvalues
                            saved_prcc_pval = np.loadtxt("./sens_data/" + prcc_pvalues + "/" + filename,
                                                         delimiter=';')
                            # load agg-prcc-pvalues
                            # saved_agg_values = np.loadtxt("./sens_data/" + agg_values + "/" +
                            #                               filename.replace("PRCC_Pvalues", "agg_values"),
                            #                               delimiter=';')
                            # Plot results
                            plot = src.Plotter(sim_obj=self)
                            plot.plot_contact_matrices_hungary(filename="contact_matrix")
                            plot.generate_prcc_p_values_heatmaps(
                                prcc_vector=abs(saved_prcc_pval[:, 0]),
                                p_values=saved_prcc_pval[:, 1],
                                filename_without_ext=filename_without_ext, target="R0")

                            # plot.plot_aggregation_prcc_pvalues(
                            #     prcc_vector=abs(saved_prcc_pval[:, 0]),
                            #     p_values=abs(saved_prcc_pval[:, 1]),
                            #     filename_without_ext=filename_without_ext)

                else:
                    # use calculated PRCC values from the previous step
                    plot = src.Plotter(sim_obj=self)
                    plot.plot_contact_matrices_hungary(filename="contact_matrx")

    def aggregate_prcc_values(self, prcc_calculator, fname):
        # save aggregated prcc values that generate values using different approach
        agg_methods = ["simple", "relN", "relM", "cm", "cmT", "cmR", "CMT", "pval"]
        for agg_typ in agg_methods:
            agg_prcc = prcc_calculator.aggregate_lockdown_approaches(
                cm=self.contact_matrix,
                agg_typ=agg_typ)
            agg_pval = prcc_calculator.aggregate_p_values_approach(
                cm=self.contact_matrix,
                agg_typ=agg_typ)
            stack_agg_prcc_pval = np.hstack([agg_prcc, agg_pval]).reshape(2, 16).T
            os.makedirs("./sens_data/agg_values", exist_ok=True)
            filename = "sens_data/agg_values" + "/" + "_".join([fname, agg_typ])
            np.savetxt(fname=filename + ".csv", X=stack_agg_prcc_pval, delimiter=";")
