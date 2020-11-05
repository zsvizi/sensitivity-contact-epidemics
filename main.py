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
susc_choices = [0.1, 0.5, 1.0]
r0_choices = [1.65, 2.2, 3]
lower_matrix_types = ["home", "normed"]
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

                    number_of_samples = 40000
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
                    filename = "./sens_data/simulations/simulation_Hungary_" + \
                               "_".join([str(susc), str(base_r0), format(beta, '.5f'), str(mtx_type)])
                    np.savetxt(fname=filename + ".csv", X=np.asarray(sim_output), delimiter=";")
                else:
                    if susc in [1.0] and base_r0 in [1.65] and mtx_type == "home":
                        # Get modified contact matrix
                        cm_row = lower_limit_matrices[mtx_type][upper_tri_indexes]

                        # Total full contact matrix contact
                        cm_full = upper_limit_matrix[upper_tri_indexes]

                        # School closure
                        contact_school = data.contact_data["school"]
                        contact_school[3, :] = contact_school[:, 3] = 0
                        full_contact_matrix = data.contact_data["home"] + data.contact_data["work"] + \
                                              contact_school + data.contact_data["other"]
                        cm_school_closure = (full_contact_matrix * age_vector)[upper_tri_indexes]

                        cm1 = get_contact_matrix_from_upper_triu(rvector=cm_full,
                                                                 age_vector=age_vector.reshape(-1, ))
                        cm2 = get_contact_matrix_from_upper_triu(rvector=cm_school_closure,
                                                                 age_vector=age_vector.reshape(-1, ))
                        cm_list = [cm1, cm2]

                        for cm in cm_list:
                            beta_cm = base_r0 / r0generator.get_eig_val(contact_mtx=cm,
                                                                        susceptibles=susceptibles.reshape(1, -1),
                                                                        population=population)[0]
                            print("R0 for modified contact matrix:", (beta / beta_cm) * base_r0)
                        # Get solution
                        t = np.linspace(0, 600, 1000)
                        plot_solution(t, params, cm_list, ["ih", "ic"], "_".join([str(susc), str(base_r0)]))


def plot_solution(time, params, cm_list, compartments, title_part):
    os.makedirs("./sens_data/dinamics", exist_ok=True)
    # for comp in compartments:
    for cm in cm_list:
        solution = model.get_solution(t=time, parameters=params, cm=cm)
        plt.plot(time, np.sum(solution[:, model.c_idx["ih"] * no_ag:(model.c_idx["ih"] + 1) * no_ag], axis=1) +
                       np.sum(solution[:, model.c_idx["icr"] * no_ag:(model.c_idx["icr"] + 1) * no_ag], axis=1))
        plt.savefig('./sens_data/dinamics/solution' + "_" + title_part + "_" + "ihicr" + '.pdf', format="pdf")
    plt.close()
    # for cm in cm_list:
    #     solution = model.get_solution(t=time, parameters=params, cm=cm)
    #     plt.plot(time, np.sum(solution[:, model.c_idx["c"] * no_ag:(model.c_idx["c"] + 1) * no_ag], axis=1))
    #     plt.savefig('./sens_data/dinamics/solution' + "_" + title_part + "_" + "c" + '.pdf', format="pdf")
    # plt.close()


if __name__ == "__main__":
    main()
    # plotter.generate_prcc_plots()
    # plotter.generate_stacked_plots()
    # plotter.plot_2d_contact_matrices()
