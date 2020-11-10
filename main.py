import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataloader import DataLoader
from model import RostModelHungary
from prcc import create_latin_table, get_contact_matrix_from_upper_triu, get_prcc_values
from r0 import R0Generator
from varname import nameof

import plotter

# Load data
data = DataLoader()

# Define initial configs
susc_choices = [0.1, 0.5, 1.0]
r0_choices = [1.65, 2.2, 3, 1.32]
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
                    if susc in [1.0, 0.5] and base_r0 in [1.32, 1.65] and mtx_type == "home":
                        # if mtx_type == "home":
                        # Get modified contact matrix
                        # cm_row = lower_limit_matrices[mtx_type][upper_tri_indexes]

                        # Total full contact matrix contact
                        cm_full = upper_limit_matrix[upper_tri_indexes]

                        # School closure
                        contact_school = np.copy(data.contact_data["school"])
                        contact_school[3, :] = 0
                        contact_school[:, 3] = 0
                        full_contact_matrix = data.contact_data["home"] + data.contact_data["work"] + \
                                              contact_school + data.contact_data["other"]
                        cm_school_closure = (full_contact_matrix * age_vector)[upper_tri_indexes]

                        # School closure + 0.5 * other
                        contact_other = np.copy(data.contact_data["other"]) * age_vector
                        contact_other[3, :] = contact_other[3, :] * 0.5
                        contact_other[:, 3] = contact_other[:, 3] * 0.5
                        contact_other[3, 3] = contact_other[3, 3] * 2

                        full_contact_matrix = data.contact_data["home"] + data.contact_data["work"] + \
                                              contact_school

                        cm_school_closure_50 = ((full_contact_matrix * age_vector) + contact_other)[upper_tri_indexes]

                        # Idosek home-on kivul nullázása
                        contact_other = np.copy(data.contact_data["other"]) * age_vector
                        contact_other[-2:, :] = 0
                        contact_other[:, -2:] = 0
                        contact_school = np.copy(data.contact_data["school"]) * age_vector
                        contact_school[-2:, :] = 0
                        contact_school[:, -2:] = 0
                        contact_work = np.copy(data.contact_data["work"]) * age_vector
                        contact_work[-2:, :] = 0
                        contact_work[:, -2:] = 0

                        cm_idosek_homeon_kivul_nulla = ((data.contact_data["home"] * age_vector) + contact_other +
                                                        contact_school + contact_work)[upper_tri_indexes]

                        # Idősek home-on kívül 0.5-szörözés
                        contact_other = np.copy(data.contact_data["other"]) * age_vector
                        contact_other[-2:, :] = contact_other[-2:, :] * 0.5
                        contact_other[:, -2:] = contact_other[:, -2:] * 0.5
                        contact_other[-2, -2] = contact_other[-2, -2] * 2
                        contact_other[-1, -1] = contact_other[-1, -1] * 2
                        contact_school = np.copy(data.contact_data["school"]) * age_vector
                        contact_school[-2:, :] = contact_school[-2:, :] * 0.5
                        contact_school[:, -2:] = contact_school[:, -2:] * 0.5
                        contact_school[-2, -2] = contact_school[-2, -2] * 2
                        contact_school[-1, -1] = contact_school[-1, -1] * 2
                        contact_work = np.copy(data.contact_data["work"]) * age_vector
                        contact_work[-2:, :] = contact_work[-2:, :] * 0.5
                        contact_work[:, -2:] = contact_work[:, -2:] * 0.5
                        contact_work[-2, -2] = contact_work[-2, -2] * 2
                        contact_work[-1, -1] = contact_work[-1, -1] * 2

                        cm_idosek_homeon_kivul_nulla_50 = \
                        ((data.contact_data["home"] * age_vector) + contact_other + contact_school + contact_work)[
                            upper_tri_indexes]

                        # Fixed number contact reduction
                        fixed_number = np.min((upper_limit_matrix - contact_home * age_vector))
                        other_than_home = upper_limit_matrix - (data.contact_data["home"] * age_vector)
                        cm_fixed_reduced = (upper_limit_matrix - fixed_number * other_than_home /
                                            np.sum(other_than_home[upper_tri_indexes]))[upper_tri_indexes]

                        # School closure + Idősek home-on kívül 0.5-szörözés
                        contact_school = np.copy(data.contact_data["school"]) * age_vector
                        contact_school[3, :] = 0
                        contact_school[:, 3] = 0

                        contact_other = np.copy(data.contact_data["other"]) * age_vector
                        contact_other[-2:, :] = contact_other[-2:, :] * 0.5
                        contact_other[:, -2:] = contact_other[:, -2:] * 0.5
                        contact_other[-2, -2] = contact_other[-2, -2] * 2
                        contact_other[-1, -1] = contact_other[-1, -1] * 2

                        contact_school[-2:, :] = 0.5 * contact_school[-2:, :]
                        contact_school[:, -2:] = 0.5 * contact_school[:, -2:]
                        contact_school[-2, -2] = 2 * contact_school[-2, -2]
                        contact_school[-1, -1] = 2 * contact_school[-1, -1]

                        contact_work = np.copy(data.contact_data["work"]) * age_vector
                        contact_work[-2:, :] = 0.5 * contact_work[-2:, :]
                        contact_work[:, -2:] = 0.5 * contact_work[:, -2:]
                        contact_work[-2, -2] = 2 * contact_work[-2, -2]
                        contact_work[-1, -1] = 2 * contact_work[-1, -1]

                        cm_kombo_1 = \
                        ((data.contact_data["home"] * age_vector) + contact_other + contact_school + contact_work)[
                            upper_tri_indexes]

                        # School closure + 50%other + Elder just home
                        contact_school = np.copy(data.contact_data["school"]) * age_vector
                        contact_school[3, :] = 0
                        contact_school[:, 3] = 0
                        contact_school[-2:, :] = 0
                        contact_school[:, -2:] = 0
                        contact_other = np.copy(data.contact_data["other"]) * age_vector
                        contact_other[3, :] = contact_other[3, :] * 0.5
                        contact_other[:, 3] = contact_other[:, 3] * 0.5
                        contact_other[3, 3] = contact_other[3, 3] * 2
                        contact_other[-2:, :] = 0
                        contact_other[:, -2:] = 0
                        contact_work = np.copy(data.contact_data["work"]) * age_vector
                        contact_work[-2:, :] = 0
                        contact_work[:, -2:] = 0

                        cm_kombo_2 = \
                        ((data.contact_data["home"] * age_vector) + contact_other + contact_school + contact_work)[
                            upper_tri_indexes]

                        # Fixed number of school closure
                        contact_school = np.copy(data.contact_data["school"]) * age_vector
                        fixed_number = np.sum(contact_school[3, :])
                        other_than_home = upper_limit_matrix - (data.contact_data["home"] * age_vector)
                        cm_fixed_reduced_2 = (upper_limit_matrix - fixed_number * other_than_home /
                                              np.sum(other_than_home[upper_tri_indexes]))[upper_tri_indexes]

                        cm1 = get_contact_matrix_from_upper_triu(rvector=cm_full,
                                                                 age_vector=age_vector.reshape(-1, ))
                        cm2 = get_contact_matrix_from_upper_triu(rvector=cm_school_closure,
                                                                 age_vector=age_vector.reshape(-1, ))
                        cm3 = get_contact_matrix_from_upper_triu(rvector=cm_school_closure_50,
                                                                 age_vector=age_vector.reshape(-1, ))
                        cm4 = get_contact_matrix_from_upper_triu(rvector=cm_idosek_homeon_kivul_nulla,
                                                                 age_vector=age_vector.reshape(-1, ))
                        cm5 = get_contact_matrix_from_upper_triu(rvector=cm_idosek_homeon_kivul_nulla_50,
                                                                 age_vector=age_vector.reshape(-1, ))
                        cm6 = get_contact_matrix_from_upper_triu(rvector=cm_fixed_reduced,
                                                                 age_vector=age_vector.reshape(-1, ))
                        cm7 = get_contact_matrix_from_upper_triu(rvector=cm_kombo_1,
                                                                 age_vector=age_vector.reshape(-1, ))
                        cm8 = get_contact_matrix_from_upper_triu(rvector=cm_kombo_2,
                                                                 age_vector=age_vector.reshape(-1, ))
                        cm9 = get_contact_matrix_from_upper_triu(rvector=cm_fixed_reduced_2,
                                                                 age_vector=age_vector.reshape(-1, ))

                        # cm_list = [cm1, cm6, cm2, cm3, cm4, cm5, cm9, cm7, cm8]
                        # legend_list = [
                        #     "Full contact matrix",
                        #     "Fixed number of contacts reduced",
                        #     "School closure",
                        #     "School closure + 50% other",
                        #     "Elder people just home",
                        #     "Elder people 1*home + 0.5 others",
                        #     "Fixed school closure number",
                        #     "School closure + Elder home-0.5*others",
                        #     "School closure + 50% other + Elder people just home"
                        # ]


                        cm_list = [cm1, cm7, cm8]
                        legend_list = [
                            "Full contact matrix",
                            "School closure + Elder home-0.5*others",
                            "School closure + 50% other + Elder people just home"
                        ]
                        r0_list = []
                        for idx, cm in enumerate(cm_list):
                            beta_cm = base_r0 / r0generator.get_eig_val(contact_mtx=cm,
                                                                        susceptibles=susceptibles.reshape(1, -1),
                                                                        population=population)[0]
                            print(str(idx) + " R0 for modified contact matrix:", (beta / beta_cm) * base_r0)
                            r0_list.append((beta / beta_cm) * base_r0)

                        # Get solution
                        t = np.linspace(0, 450, 1000)
                        plot_solution(t, params, cm_list, legend_list, "_V3_".join([str(susc), str(base_r0)]))

                        cm_list = [cm1, cm6, cm4, cm5]
                        legend_list = [
                            "Full contact matrix",
                            "Fixed number of contacts reduced",
                            "Elder people just home",
                            "Elder people 1*home + 0.5 others"
                        ]
                        r0_list = []
                        for idx, cm in enumerate(cm_list):
                            beta_cm = base_r0 / r0generator.get_eig_val(contact_mtx=cm,
                                                                        susceptibles=susceptibles.reshape(1, -1),
                                                                        population=population)[0]
                            print(str(idx) + " R0 for modified contact matrix:", (beta / beta_cm) * base_r0)
                            r0_list.append((beta / beta_cm) * base_r0)

                        # Get solution
                        t = np.linspace(0, 450, 1000)
                        plot_solution(t, params, cm_list, legend_list, "_V2_".join([str(susc), str(base_r0)]))

                        cm_list = [cm1, cm2, cm3, cm9]
                        legend_list = [
                            "Full contact matrix",
                            "School closure",
                            "School closure + 50% other",
                            "Fixed school closure number"
                        ]
                        r0_list = []
                        for idx, cm in enumerate(cm_list):
                            beta_cm = base_r0 / r0generator.get_eig_val(contact_mtx=cm,
                                                                        susceptibles=susceptibles.reshape(1, -1),
                                                                        population=population)[0]
                            print(str(idx) + " R0 for modified contact matrix:", (beta / beta_cm) * base_r0)
                            r0_list.append((beta / beta_cm) * base_r0)

                        # Get solution
                        t = np.linspace(0, 450, 1000)
                        plot_solution(t, params, cm_list, legend_list, "_V1_".join([str(susc), str(base_r0)]))


def plot_solution(time, params, cm_list, legend_list, title_part):
    os.makedirs("./sens_data/dinamics", exist_ok=True)
    # for comp in compartments:
    for idx, cm in enumerate(cm_list):
        solution = model.get_solution(t=time, parameters=params, cm=cm)
        plt.plot(time, np.sum(solution[:, model.c_idx["ic"] * no_ag:(model.c_idx["ic"] + 1) * no_ag], axis=1),
                 label=legend_list[idx])
        plt.legend(loc="upper left")
    plt.gca().set_xlabel('days')
    plt.gca().set_ylabel('ICU usage')
    plt.tight_layout()
    plt.savefig('./sens_data/dinamics/solution' + "_" + title_part + "_" + "ic" + '.pdf', format="pdf")
    plt.close()


if __name__ == "__main__":
    main()
    # plotter.generate_prcc_plots()
    # plotter.generate_stacked_plots()
    # plotter.plot_2d_contact_matrices()
