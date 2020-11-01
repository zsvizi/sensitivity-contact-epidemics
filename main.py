import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataloader import DataLoader
from model import RostModelHungary
from prcc import create_latin_table, get_contact_matrix_from_upper_triu, get_prcc_values
from r0 import R0Generator

import plotter

# Load data
data = DataLoader()

# Define initial configs
susc_choices = [0.5]  # [0.1, 0.5, 1.0]
r0_choices = [2.2]  # [1.65, 2.2, 3]
lower_matrix_types = ["home"]  # ["home", "normed"]
no_ag = data.contact_data["home"].shape[0]
model = RostModelHungary(model_data=data)
population = model.population
age_vector = population.reshape((-1, 1))
susceptibles = model.get_initial_values()[model.c_idx["s"] * no_ag:(model.c_idx["s"] + 1) * no_ag]

contact_matrix = data.contact_data["home"] + data.contact_data["work"] + \
                 data.contact_data["school"] + data.contact_data["other"]
contact_home = data.contact_data["home"]
upper_limit_matrix = contact_matrix * age_vector
lower_limit_matrices = {"normed": upper_limit_matrix - np.min(upper_limit_matrix - contact_home * age_vector),
                        "home": contact_home * age_vector}
upper_tri_indexes = np.triu_indices(no_ag)


def main():
    is_lhs_generated = False
    # Instantiate model
    # ic_idx = model.c_idx["ic"]

    # Get contact matrices


    # --- Get complete parameters dictionary ---
    # 0. Get base parameter dictionary
    params = data.model_parameters_data
    # 1. Update params by susceptibility vector
    susceptibility = np.ones(no_ag)
    for susc in susc_choices:
        susceptibility[:4] = susc
        params.update({"susc": susceptibility})
        # 2. Update params by calculated BASELINE beta
        for base_r0 in r0_choices:
            r0generator = R0Generator(param=params)
            beta = base_r0 / r0generator.get_eig_val(contact_mtx=contact_matrix,
                                                     susceptibles=susceptibles.reshape(1, -1),
                                                     population=population)[0]
            params.update({"beta": beta})
            # 3. Choose matrix type
            for mtx_type in lower_matrix_types:
                if is_lhs_generated:
                    lower_limit_matrix = lower_limit_matrices[mtx_type]

                    number_of_samples = 10
                    lhs_table = create_latin_table(n_of_samples=number_of_samples,
                                                   lower=lower_limit_matrix[upper_tri_indexes],
                                                   upper=upper_limit_matrix[upper_tri_indexes])
                    print("Simulation for", number_of_samples,
                          "samples (", "-".join([str(susc), str(base_r0), mtx_type]), ")")
                    sleep(0.3)

                    def get_lhs_output(cm_rvector):
                        cm_ndarray = get_contact_matrix_from_upper_triu(rvector=cm_rvector,
                                                                        age_vector=age_vector.reshape(-1, ))
                        beta_lhs = base_r0 / r0generator.get_eig_val(contact_mtx=cm_ndarray,
                                                                     susceptibles=susceptibles.reshape(1, -1),
                                                                     population=population)[0]
                        r0_lhs = (beta / beta_lhs) * base_r0
                        output = np.append(cm_rvector, [0, r0_lhs])
                        output = np.append(output, np.zeros(no_ag))
                        return list(output)

                    results = list(tqdm(map(get_lhs_output, lhs_table), total=lhs_table.shape[0]))
                    results = np.array(results)
                    r0_col_idx = int((no_ag + 1) * no_ag / 2 + 1)
                    results = results[results[:, r0_col_idx].argsort()]
                    sim_output = np.array(results)
                    sleep(0.3)

                    os.makedirs("./sens_data", exist_ok=True)
                    os.makedirs("./sens_data/simulations", exist_ok=True)
                    filename = "./sens_data/simulations/simulation_" +\
                               "_".join([str(susc), str(base_r0), format(beta, '.5f'), str(mtx_type)])
                    np.savetxt(fname=filename + ".csv", X=np.asarray(sim_output), delimiter=";")
                else:
                    # Get modified contact matrix
                    cm_row = lower_limit_matrices[mtx_type][upper_tri_indexes]
                    cm = get_contact_matrix_from_upper_triu(rvector=cm_row,
                                                            age_vector=age_vector.reshape(-1, ))
                    beta_cm = base_r0 / r0generator.get_eig_val(contact_mtx=cm,
                                                                susceptibles=susceptibles.reshape(1, -1),
                                                                population=population)[0]
                    print("R0 for modified contact matrix:", (beta / beta_cm) * base_r0)
                    # Get solution
                    t = np.linspace(0, 200, 500)
                    plot_solution(t, params, cm, ["ih"])


def plot_solution(time, params, cm, compartments):
    os.makedirs("./sens_data", exist_ok=True)
    for comp in compartments:
        solution = model.get_solution(t=time, parameters=params, cm=cm)
        plt.plot(time, np.sum(solution[:, model.c_idx[comp] * no_ag:(model.c_idx[comp] + 1) * no_ag], axis=1))
        plt.savefig('./sens_data/solution' + "_" + comp + '.pdf', format="pdf")


if __name__ == "__main__":
    main()
    # plotter.generate_prcc_plots()
    # plotter.generate_stacked_plots()
    # plotter.plot_2d_contact_matrices()