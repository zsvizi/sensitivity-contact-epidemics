import os
from time import sleep

import numpy as np
from tqdm import tqdm

from dataloader import DataLoader
from model import RostModelHungary
from prcc import create_latin_table, get_contact_matrix_from_upper_triu
from r0 import R0Generator


class Simulation:
    def __init__(self):
        # Load data
        self.data = DataLoader()

        # User-defined parameters
        self.susc_choices = [0.1, 0.5, 1.0]
        self.r0_choices = [1.35]
        self.lower_matrix_types = ["home", "normed"]

        # Define initial configs
        self._get_initial_config()

        # User-defined lower matrix types
        self.lower_limit_matrices = \
            {"normed": self.upper_limit_matrix - np.min(self.upper_limit_matrix - self.contact_home * self.age_vector),
             "home": self.contact_home * self.age_vector}

    def run(self):
        is_lhs_generated = False

        # 0. Get base parameter dictionary
        params = self.data.model_parameters_data
        # 1. Update params by susceptibility vector
        susceptibility = np.ones(self.no_ag)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            params.update({"susc": susceptibility})
            # 2. Update params by calculated BASELINE beta
            for base_r0 in self.r0_choices:
                r0generator = R0Generator(param=params)
                beta = base_r0 / r0generator.get_eig_val(contact_mtx=self.contact_matrix,
                                                         susceptibles=self.susceptibles.reshape(1, -1),
                                                         population=self.population)[0]
                params.update({"beta": beta})
                # 3. Choose matrix type
                for mtx_type in self.lower_matrix_types:
                    if is_lhs_generated:
                        self.generate_lhs(base_r0, beta, mtx_type, susc, r0generator)

    def generate_lhs(self, base_r0, beta, mtx_type, susc, r0generator):
        # Get actual lower limit matrix
        lower_limit_matrix = self.lower_limit_matrices[mtx_type]

        # Get LHS tables
        number_of_samples = 40000
        lhs_table = create_latin_table(n_of_samples=number_of_samples,
                                       lower=lower_limit_matrix[self.upper_tri_indexes],
                                       upper=self.upper_limit_matrix[self.upper_tri_indexes])
        print("Simulation for", number_of_samples,
              "samples (", "-".join([str(susc), str(base_r0), mtx_type]), ")")
        sleep(0.3)

        # Local function for calculating LHS output
        def get_lhs_output(cm_rvector):
            cm_ndarray = get_contact_matrix_from_upper_triu(rvector=cm_rvector,
                                                            age_vector=self.age_vector.reshape(-1, ))
            beta_lhs = base_r0 / r0generator.get_eig_val(contact_mtx=cm_ndarray,
                                                         susceptibles=self.susceptibles.reshape(1, -1),
                                                         population=self.population)[0]
            r0_lhs = (beta / beta_lhs) * base_r0
            output = np.append(cm_rvector, [0, r0_lhs])
            output = np.append(output, np.zeros(self.no_ag))
            return list(output)

        # Generate LHS output
        results = list(tqdm(map(get_lhs_output, lhs_table), total=lhs_table.shape[0]))
        results = np.array(results)

        # Append R0 values
        r0_col_idx = int((self.no_ag + 1) * self.no_ag / 2 + 1)
        results = results[results[:, r0_col_idx].argsort()]
        sim_output = np.array(results)
        sleep(0.3)

        # Save LHS output
        os.makedirs("./sens_data", exist_ok=True)
        os.makedirs("./sens_data/simulations", exist_ok=True)
        filename = "./sens_data/simulations/simulation_Hungary_" + \
                   "_".join([str(susc), str(base_r0), format(beta, '.5f'), str(mtx_type)])
        np.savetxt(fname=filename + ".csv", X=np.asarray(sim_output), delimiter=";")

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
        self.upper_limit_matrix = self.contact_matrix * self.age_vector
        self.upper_tri_indexes = np.triu_indices(self.no_ag)


def main():
    simulation = Simulation()
    simulation.run()


if __name__ == '__main__':
    main()
