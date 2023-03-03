import os
import numpy as np
from src.prcc import get_prcc_values, get_rectangular_matrix_from_upper_triu


class PRCCalculator:
    def __init__(self, n_ag, age_vector, params):

        self.n_ag = n_ag
        self.age_vector = age_vector
        self.params = params

        self.upp_tri_size = int((self.n_ag + 1) * self.n_ag / 2)

        self.prcc_values = dict()
        self.prcc_mtx = []

        self.calculate_prcc_values()

    def calculate_prcc_values(self):
        sim_folder, lhs_folder = ["simulations", "lhs"]

        for root, dirs, files in os.walk("./sens_data/" + sim_folder):
            for filename in files:
                filename_without_ext = os.path.splitext(filename)[0]
                print(filename_without_ext)
                saved_simulation = np.loadtxt("./sens_data/" + sim_folder + "/" + filename,
                                              delimiter=';')
                saved_lhs_values = np.loadtxt(
                    "./sens_data/" + lhs_folder + "/" + filename.replace("simulations", "lhs"),
                    delimiter=';')
                if 'lockdown_3' in filename_without_ext:
                    sim_data = saved_lhs_values[:, :3 * self.upp_tri_size]
                    # Transform sim_data to get positively correlating variables
                    # Here ratio to subtract is negatively correlated to the targets, thus
                    # 1 - ratio (i.e. ratio of remaining contacts) is positively correlated
                    sim_data = 1 - sim_data

                elif "lockdown" in filename_without_ext:
                    sim_data = saved_lhs_values[:, :(self.n_ag * (self.n_ag + 1)) // 2]
                    sim_data = 1 - sim_data

                elif "mitigation" in filename_without_ext:
                    sim_data = saved_lhs_values[:, :(self.n_ag * (self.n_ag + 1)) // 2]

                else:
                    raise Exception('Matrix type is unknown!')

                # PRCC analysis for R0
                simulation = np.append(sim_data, saved_simulation[:, -self.n_ag - 1].reshape((-1, 1)), axis=1)
                prcc_list = get_prcc_values(simulation)
                if 'lockdown_3' in filename_without_ext:
                    prcc_matrix_school, prcc_matrix_work, prcc_matrix_other = [
                        get_rectangular_matrix_from_upper_triu(prcc_list[:self.upp_tri_size],
                                                               self.n_ag),
                        get_rectangular_matrix_from_upper_triu(
                            prcc_list[self.upp_tri_size:2 * self.upp_tri_size],
                            self.n_ag),
                        get_rectangular_matrix_from_upper_triu(prcc_list[2 * self.upp_tri_size:],
                                                               self.n_ag)]
                    agg_prcc_school, agg_prcc_work, agg_prcc_other = [
                        (np.sum(prcc_matrix_school * self.age_vector, axis=1) /
                         np.sum(self.age_vector)).flatten(),
                        (np.sum(prcc_matrix_work * self.age_vector, axis=1) /
                         np.sum(self.age_vector)).flatten(),
                        (np.sum(prcc_matrix_other * self.age_vector, axis=1) /
                         np.sum(self.age_vector)).flatten()]
                    prcc_list = np.array([agg_prcc_school, agg_prcc_work, agg_prcc_other]).flatten()

                    self.prcc_values.update(
                        {
                            "prcc_matrix_school": prcc_matrix_school,
                            "prcc_matrix_work": prcc_matrix_work,
                            "prcc_matrix_other": prcc_matrix_other,
                            "agg_prcc_school": agg_prcc_school,
                            "agg_prcc_work": agg_prcc_work,
                            "agg_prcc_other": agg_prcc_other,
                            "prcc_list": prcc_list
                        }
                    )

                elif 'lockdown' in filename_without_ext or 'mitigation' in filename_without_ext:
                    prcc_mtx = get_rectangular_matrix_from_upper_triu(prcc_list[:self.upp_tri_size],
                                                                      self.n_ag)
                    self.prcc_mtx = prcc_mtx
                    p_icr = (1 - self.params['p']) * self.params['h'] * self.params['xi']
                    print(p_icr)

                # PRCC analysis for ICU maximum
                simulation_2 = np.append(sim_data, saved_simulation[:, -self.n_ag - 2].reshape((-1, 1)),
                                         axis=1)
                prcc_list_2 = get_prcc_values(simulation_2)
                if 'lockdown_3' in filename_without_ext:
                    prcc_matrix_school_2 = get_rectangular_matrix_from_upper_triu(
                        prcc_list_2[: self.upp_tri_size], self.n_ag)
                    prcc_matrix_work_2 = get_rectangular_matrix_from_upper_triu(
                        prcc_list_2[self.upp_tri_size:2 * self.upp_tri_size],
                        self.n_ag)
                    prcc_matrix_other_2 = get_rectangular_matrix_from_upper_triu(
                        prcc_list_2[2 * self.upp_tri_size:], self.n_ag)

                    agg_prcc_school_2 = \
                        (np.sum(prcc_matrix_school_2 * self.age_vector, axis=1) /
                         np.sum(self.age_vector)).flatten()
                    agg_prcc_work_2 = \
                        (np.sum(prcc_matrix_work_2 * self.age_vector, axis=1) /
                         np.sum(self.age_vector)).flatten()
                    agg_prcc_other_2 = \
                        (np.sum(prcc_matrix_other_2 * self.age_vector, axis=1) /
                         np.sum(self.age_vector)).flatten()

                    prcc_list_2 = np.array([agg_prcc_school_2, agg_prcc_work_2, agg_prcc_other_2]).flatten()

                    self.prcc_values.update(
                        {
                            "prcc_matrix_school_2": prcc_matrix_school_2,
                            "prcc_matrix_work_2": prcc_matrix_work_2,
                            "prcc_matrix_other_2": prcc_matrix_other_2,
                            "agg_prcc_school_2": agg_prcc_school_2,
                            "agg_prcc_work_2": agg_prcc_work_2,
                            "agg_prcc_other_2": agg_prcc_other_2,
                            "prcc_list_2": prcc_list_2
                        }
                    )

                elif 'lockdown' in filename_without_ext or 'mitigation' in filename_without_ext:
                    prcc_mtx = get_rectangular_matrix_from_upper_triu(prcc_list[:self.upp_tri_size],
                                                                      self.n_ag)
                    p_icr = (1 - self.params['p']) * self.params['h'] * self.params['xi']
                    # store prcc and aggregate values
                    self.prcc_values.update(
                        {
                            "prcc_mtx": prcc_mtx,
                            "p_icr": p_icr
                        }
                    )
