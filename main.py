import os
from time import sleep

import numpy as np
from tqdm import tqdm

from analysis import Analysis
from dataloader import DataLoader
from model import RostModelHungary
from plotter import generate_prcc_plots
from prcc import create_latin_table
from r0 import R0Generator


class Simulation:
    def __init__(self):
        # Load data
        self.data = DataLoader()

        # User-defined parameters
        self.susc_choices = [0.1, 0.5, 1.0]
        self.r0_choices = [1.35]

        # Define initial configs
        self._get_initial_config()

        # User-defined lower matrix types
        self.lhs_boundaries = \
            {"unit": {"lower": np.zeros(self.no_ag),
                      "upper": np.ones(self.no_ag) * self._get_upper_bound_factor_unit()}
             }
        self.lower_matrix_types = list(self.lhs_boundaries.keys())

    def run(self):
        is_lhs_generated = False
        is_prcc_plots_generated = True

        # 1. Update params by susceptibility vector
        susceptibility = np.ones(self.no_ag)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            self.params.update({"susc": susceptibility})
            # 2. Update params by calculated BASELINE beta
            for base_r0 in self.r0_choices:
                r0generator = R0Generator(param=self.params)
                beta = base_r0 / r0generator.get_eig_val(contact_mtx=self.contact_matrix,
                                                         susceptibles=self.susceptibles.reshape(1, -1),
                                                         population=self.population)[0]
                self.params.update({"beta": beta})
                # 3. Choose matrix type
                for mtx_type in self.lower_matrix_types:
                    if is_lhs_generated:
                        self.generate_lhs(base_r0, beta, mtx_type, susc, r0generator)
                    else:
                        if susc in [1.0, 0.5] and base_r0 in [1.35] and mtx_type == "home":
                            analysis = Analysis(sim=self, susc=susc, base_r0=base_r0, mtx_type=mtx_type)
                            analysis.run()
        if is_prcc_plots_generated:
            generate_prcc_plots(sim_obj=self)

    def generate_lhs(self, base_r0, beta, mtx_type, susc, r0generator):
        # Get actual lower limit matrix
        lower_bound = self.lhs_boundaries[mtx_type]["lower"]
        upper_bound = self.lhs_boundaries[mtx_type]["upper"]

        # Get LHS tables
        number_of_samples = 40000
        lhs_table = create_latin_table(n_of_samples=number_of_samples,
                                       lower=lower_bound,
                                       upper=upper_bound)
        print("Simulation for", number_of_samples,
              "samples (", "-".join([str(susc), str(base_r0), mtx_type]), ")")
        sleep(0.3)

        # Local function for calculating LHS output
        def get_simulation_output(lhs_sample):
            cm_total_sim = self.contact_matrix * self.age_vector - lhs_sample.reshape(-1, 1)
            cm_sim = cm_total_sim / self.age_vector
            beta_lhs = base_r0 / r0generator.get_eig_val(contact_mtx=cm_sim,
                                                         susceptibles=self.susceptibles.reshape(1, -1),
                                                         population=self.population)[0]
            r0_lhs = (beta / beta_lhs) * base_r0
            output = np.append(cm_total_sim[self.upper_tri_indexes], [0, r0_lhs])
            output = np.append(output, np.zeros(self.no_ag))
            return list(output)

        # Generate LHS output
        results = list(tqdm(map(get_simulation_output, lhs_table), total=lhs_table.shape[0]))
        results = np.array(results)

        # Sort tables by R0 values
        r0_col_idx = int((self.no_ag + 1) * self.no_ag / 2 + 1)
        sorted_idx = results[:, r0_col_idx].argsort()
        results = results[sorted_idx]
        lhs_table = np.array(lhs_table[sorted_idx])
        sim_output = np.array(results)
        sleep(0.3)

        # Create directories for saving calculation outputs
        os.makedirs("./sens_data", exist_ok=True)
        os.makedirs("./sens_data/simulations", exist_ok=True)
        os.makedirs("./sens_data/lhs", exist_ok=True)
        # Save simulation input
        filename = "./sens_data/simulations/simulation_Hungary_" + \
                   "_".join([str(susc), str(base_r0), format(beta, '.5f'), str(mtx_type)])
        np.savetxt(fname=filename + ".csv", X=np.asarray(sim_output), delimiter=";")
        # Save LHS output
        filename = "./sens_data/lhs/lhs_Hungary_" + \
                   "_".join([str(susc), str(base_r0), format(beta, '.5f'), str(mtx_type)])
        np.savetxt(fname=filename + ".csv", X=np.asarray(lhs_table), delimiter=";")

    def _get_initial_config(self):
        self.no_ag = self.data.contact_data["home"].shape[0]
        self.model = RostModelHungary(model_data=self.data)
        self.population = self.model.population
        self.age_vector = self.population.reshape((-1, 1))
        self.susceptibles = self.model.get_initial_values()[self.model.c_idx["s"] *
                                                            self.no_ag:(self.model.c_idx["s"] + 1) * self.no_ag]
        self.contact_matrix = self.data.contact_data["home"] + self.data.contact_data["work"] + \
            self.data.contact_data["school"] + self.data.contact_data["other"]
        self.contact_home = self.data.contact_data["home"]
        self.upper_tri_indexes = np.triu_indices(self.no_ag)
        # 0. Get base parameter dictionary
        self.params = self.data.model_parameters_data

    def _get_upper_bound_factor_unit(self):
        cm_diff = (self.contact_matrix - self.contact_home) * self.age_vector
        min_diff = self.no_ag * np.min(cm_diff) / 2
        return min_diff


def main():
    simulation = Simulation()
    simulation.run()


if __name__ == '__main__':
    main()
