import os
import numpy as np
from prcc import get_prcc_values, get_rectangular_matrix_from_upper_triu
import prcc


class PRCCalculator:
    def __init__(self, sim_obj):
        self.sim_obj = sim_obj
        self.prcc_values = dict()
        self.calculate_prcc_values()

    def calculate_prcc_values(self):
        sim_folder, lhs_folder = ["simulations", "lhs"]

        for root, dirs, files in os.walk("./sens_data/" + sim_folder):
            for filename in files:
                filename_without_ext = os.path.splitext(filename)[0]
                saved_simulation = np.loadtxt("./sens_data/" + sim_folder + "/" + filename,
                                              delimiter=';')
                saved_lhs_values = np.loadtxt(
                    "./sens_data/" + lhs_folder + "/" + filename.replace("simulations", "lhs"),
                    delimiter=';')
                if 'lockdown_3' in filename_without_ext:
                    sim_data = saved_lhs_values[:, :3 * self.sim_obj.upp_tri_size]
                    # Transform sim_data to get positively correlating variables
                    # Here ratio to subtract is negatively correlated to the targets, thus
                    # 1 - ratio (i.e. ratio of remaining contacts) is positively correlated
                    sim_data = 1 - sim_data
                    names = [str(i) for i in range(3 * self.sim_obj.n_ag)]
                elif "lockdown" in filename_without_ext:
                    sim_data = saved_lhs_values[:, :(self.sim_obj.n_ag * (self.sim_obj.n_ag + 1)) // 2]
                    sim_data = 1 - sim_data

                elif "mitigation" in filename_without_ext:
                    sim_data = saved_lhs_values[:, :(self.sim_obj.n_ag * (self.sim_obj.n_ag + 1)) // 2]

                else:
                    raise Exception('Matrix type is unknown!')

                # PRCC analysis for R0
                simulation = np.append(sim_data, saved_simulation[:, -self.sim_obj.n_ag - 1].reshape((-1, 1)), axis=1)
                prcc_list = get_prcc_values(simulation)
                if 'lockdown_3' in filename_without_ext:
                    prcc_matrix_school, prcc_matrix_work, prcc_matrix_other = [
                        get_rectangular_matrix_from_upper_triu(prcc_list[:self.sim_obj.upp_tri_size],
                                                               self.sim_obj.n_ag),
                        get_rectangular_matrix_from_upper_triu(
                            prcc_list[self.sim_obj.upp_tri_size:2 * self.sim_obj.upp_tri_size], self.sim_obj.n_ag),
                        get_rectangular_matrix_from_upper_triu(prcc_list[2 * self.sim_obj.upp_tri_size:],
                                                               self.sim_obj.n_ag)]

                    agg_prcc_school, agg_prcc_work, agg_prcc_other = [
                        (np.sum(prcc_matrix_school * self.sim_obj.age_vector, axis=1) /
                         np.sum(self.sim_obj.age_vector)).flatten(),
                        (np.sum(prcc_matrix_work * self.sim_obj.age_vector, axis=1) /
                         np.sum(self.sim_obj.age_vector)).flatten(),
                        (np.sum(prcc_matrix_other * self.sim_obj.age_vector, axis=1) /
                         np.sum(self.sim_obj.age_vector)).flatten()]
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
                    prcc_mtx = prcc.get_rectangular_matrix_from_upper_triu(prcc_list[:self.sim_obj.upp_tri_size],
                                                                           self.sim_obj.n_ag)
                    p_icr = (1 - self.sim_obj.sim_obj.params['p']) * self.sim_obj.params['h'] * \
                            self.sim_obj.params['xi']

                # PRCC analysis for ICU maximum
                simulation_2 = np.append(sim_data, saved_simulation[:, -self.sim_obj.n_ag - 2].reshape((-1, 1)),
                                         axis=1)
                prcc_list_2 = prcc.get_prcc_values(simulation_2)
                if 'lockdown_3' in filename_without_ext:
                    prcc_matrix_school_2 = prcc.get_rectangular_matrix_from_upper_triu(
                        prcc_list_2[: self.sim_obj.upp_tri_size], self.sim_obj.n_ag)
                    prcc_matrix_work_2 = prcc.get_rectangular_matrix_from_upper_triu(
                        prcc_list_2[self.sim_obj.upp_tri_size:2 * self.sim_obj.upp_tri_size],
                        self.sim_obj.n_ag)
                    prcc_matrix_other_2 = prcc.get_rectangular_matrix_from_upper_triu(
                        prcc_list_2[2 * self.sim_obj.upp_tri_size:], self.sim_obj.n_ag)

                    agg_prcc_school_2 = \
                        (np.sum(prcc_matrix_school_2 * self.sim_obj.age_vector, axis=1) /
                         np.sum(self.sim_obj.age_vector)).flatten()
                    agg_prcc_work_2 = \
                        (np.sum(prcc_matrix_work_2 * self.sim_obj.age_vector, axis=1) /
                         np.sum(self.sim_obj.age_vector)).flatten()
                    agg_prcc_other_2 = \
                        (np.sum(prcc_matrix_other_2 * self.sim_obj.age_vector, axis=1) /
                         np.sum(self.sim_obj.age_vector)).flatten()

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
                    prcc_mtx = prcc.get_rectangular_matrix_from_upper_triu(prcc_list[:self.sim_obj.upp_tri_size],
                                                                           self.sim_obj.n_ag)
                    p_icr = (1 - self.sim_obj.params['p']) * self.sim_obj.params['h'] * self.sim_obj.params['xi']
                    # store prcc and aggregate values
                    self.prcc_values.update(
                        {
                            "prcc_mtx": prcc_mtx,
                            "p_icr": p_icr
                        }
                    )


