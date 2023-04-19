import numpy as np
import os

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
        self.mtx_types = ["lockdown", "lockdown_3"]

        self.lhs_table = None
        self.sim_output = None
        self.prcc_values = None

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
                beta = base_r0 / r0generator.get_eig_val(contact_mtx=self.contact_matrix,
                                                         susceptibles=self.susceptibles.reshape(1, -1),
                                                         population=self.population)[0]
                self.params.update({"beta": beta})
                # 3. Choose matrix type
                for mtx_type in self.mtx_types:
                    self.sim_state.update(
                        {"base_r0": base_r0,
                         "beta": beta,
                         "type": mtx_type,
                         "susc": susc,
                         "r0generator": r0generator})
                    sampler_npi = src.sampling.sampler_npi.SamplerNPI(
                        sim_state=self.sim_state, sim_obj=self, mtx_type=mtx_type)
                    self.lhs_table, self.sim_output = sampler_npi.run()

    def calculate_prcc_values(self):
        for susc in self.susc_choices:
            for base_r0 in self.r0_choices:
                for mtx_type in self.mtx_types:
                    print(susc, base_r0, mtx_type)
                    if self.lhs_table is None:
                        # read files from the generated folder based on the given parameters
                        sim_folder, lhs_folder = "simulations", "lhs"
                        for root, dirs, files in os.walk("./sens_data/" + sim_folder):
                            for filename in files:
                                filename_without_ext = os.path.splitext(filename)[0]
                                saved_simulation = np.loadtxt("./sens_data/" + sim_folder + "/" +
                                                              filename, delimiter=';')
                                saved_lhs_values = np.loadtxt("./sens_data/" + lhs_folder + "/" +
                                                              filename.replace("simulations", "lhs"), delimiter=';')
                                # for lockdown_3 replace lockdown with lockdown_3
                                if "lockdown" in filename_without_ext:
                                    prcc_calculator = src.prcc_calculator.PRCCCalculator(number_of_samples=120000,
                                                                                         sim_obj=self)
                                    lockdown_prcc = prcc_calculator.calculate_prcc_values(mtx_typ="lockdown",
                                                                                          lhs_table=saved_lhs_values,
                                                                                          sim_output=saved_simulation)
                                    cm = np.loadtxt("sens_data/cm/cm.csv", delimiter=';')
                                    agg_methods = ["simple", "relN", "relM", "cm", "cmT", "cmR", "CMT", "pval"]
                                    for agg_typ in agg_methods:
                                        prcc_calculator.aggregate_lockdown_approaches(cm=cm, agg_typ=agg_typ,
                                                                                      mtx_typ="lockdown")
                                        os.makedirs("./sens_data/agg_values", exist_ok=True)
                                        filename = "sens_data/agg_values" + "/" + "_".join([str(susc), str(base_r0),
                                                                                           "lockdown", agg_typ])
                                        # save aggregated prcc values
                                        np.savetxt(fname=filename + ".csv", X=prcc_calculator.agg_prcc, delimiter=";")
                                        print(filename_without_ext, lockdown_prcc.shape)
                                    else:
                                        print("Matrix type lockdown_3: work & other")
                    else:
                        if "lockdown" == mtx_type:
                            prcc_calculator = src.prcc_calculator.PRCCCalculator(number_of_samples=120000,
                                                                                 sim_obj=self)
                            prcc = prcc_calculator.calculate_prcc_values(mtx_typ=mtx_type, lhs_table=self.lhs_table,
                                                                         sim_output=self.sim_output)
                            # calculate p-values
                            p_values = prcc_calculator.calculate_p_values(mtx_typ=mtx_type)
                            # save prcc and p values
                            os.makedirs("./sens_data/PRCC", exist_ok=True)
                            filename = "sens_data/PRCC" + "/" + "_".join([str(susc), str(base_r0), mtx_type])
                            x = np.hstack([prcc, prcc_calculator.p_value]).reshape(2, 136).T
                            np.savetxt(fname=filename + ".csv", X=prcc, delimiter=";")

    def plot_prcc_values(self):
        for susc in self.susc_choices:
            for base_r0 in self.r0_choices:
                for mtx_type in self.mtx_types:
                    if self.prcc_values is None:
                        print(susc, base_r0, mtx_type)
                        # read files from the generated folder based on the given parameters
                        prc_folder = "PRCC"
                        for root, dirs, files in os.walk("./sens_data/" + prc_folder):
                            for filename in files:
                                filename_without_ext = os.path.splitext(filename)[0]
                                saved_prcc = np.loadtxt("./sens_data/" + prc_folder + "/" + filename, delimiter=';')
                                # load saved aggregated prcc values
                                # saved_p = np.loadtxt("sens_data/agg_lock3/simple.csv", delimiter=';')
                                plot = src.plotter.Plotter(sim_obj=self)
                                plot.generate_prcc_plots(prcc_vector=saved_prcc,
                                                         filename_without_ext=filename_without_ext)
                                plot.plot_2d_contact_matrices()
                                plot.generate_stacked_plots()
                                plot.plot_contact_matrix_as_grouped_bars()
                    else:
                        # use calculated PRCC values from the previous step
                        plot = src.plotter.Plotter(sim_obj=self)
                        plot.plot_contact_matrix_as_grouped_bars()
                        plot.generate_stacked_plots()
                        plot.plot_2d_contact_matrices()

    def _get_upper_bound_factor_unit(self):
        cm_diff = (self.contact_matrix - self.contact_home) * self.age_vector
        min_diff = np.min(cm_diff) / 2
        return min_diff
