import os
import numpy as np
import scipy.stats as ss

from src.prcc import get_prcc_values, get_rectangular_matrix_from_upper_triu


class PRCCCalculator:
    def __init__(self, n_ag: int, age_vector: np.ndarray, params: dict):

        self.n_ag = n_ag
        self.age_vector = age_vector
        self.params = params

        self.upp_tri_size = int((self.n_ag + 1) * self.n_ag / 2)

        self.sim_data = np.array([])

        self.prcc_values = dict()
        self.prcc_mtx = np.array([])
        self.p_icr = []

        self.prcc_matrix_school = []  # 16 * 16
        self.prcc_matrix_work = []
        self.prcc_matrix_other = []

        self.prcc_matrix_school_2 = []  # 16 * 16
        self.prcc_matrix_work_2 = []
        self.prcc_matrix_other_2 = []

        # aggregated values using p-values
        self.prcc_p_values = dict()

        # p-values in the 3 settings
        self.p_values = dict()  # 16 p-values corresponding to each age group

        self.calculate_prcc_values()

    def calculate_prcc_values(self):
        sim_folder, lhs_folder = ["simulations", "lhs"]

        for root, dirs, files in os.walk("./sens_data/" + sim_folder):
            for filename in files:
                filename_without_ext = os.path.splitext(filename)[0]
                print(filename_without_ext)   # name of simulations
                saved_simulation = np.loadtxt("./sens_data/" + sim_folder + "/" + filename, delimiter=';')
                saved_lhs_values = np.loadtxt("./sens_data/" + lhs_folder + "/" +
                                              filename.replace("simulations", "lhs"), delimiter=';')
                if 'lockdown_3' in filename_without_ext:
                    sim_data = saved_lhs_values[:, :3 * self.upp_tri_size]
                    # Transform sim_data to get positively correlating variables
                    # Here ratio to subtract is negatively correlated to the targets, thus
                    # 1 - ratio (i.e. ratio of remaining contacts) is positively correlated
                    sim_data = 1 - sim_data
                elif "lockdown" in filename_without_ext:
                    sim_data = saved_lhs_values[:, :(self.n_ag * (self.n_ag + 1)) // 2]
                    sim_data = 1 - sim_data
                # elif "mitigation" in filename_without_ext:
                #     sim_data = saved_lhs_values[:, :(self.n_ag * (self.n_ag + 1)) // 2]
                else:
                    raise Exception('Matrix type is unknown!')

                # PRCC analysis for R0
                simulation = np.append(sim_data, saved_simulation[:, -self.n_ag - 1].reshape((-1, 1)), axis=1)
                prcc_list = get_prcc_values(simulation)
                if 'lockdown_3' in filename_without_ext:
                    prcc_matrix_school, prcc_matrix_work, prcc_matrix_other = [
                        get_rectangular_matrix_from_upper_triu(prcc_list[:self.upp_tri_size], self.n_ag),
                        get_rectangular_matrix_from_upper_triu(
                            prcc_list[self.upp_tri_size:2 * self.upp_tri_size], self.n_ag),
                        get_rectangular_matrix_from_upper_triu(prcc_list[2 * self.upp_tri_size:], self.n_ag)]

                    # calculate correlation (r) and p-values (p). Returns (16 * 16) p-values matrix for each age group
                    rows, cols = prcc_matrix_school.shape
                    [school_r, school_p] = [np.ones((cols, cols)),  np.ones((cols, cols))]
                    [work_r, work_p] = [np.ones((cols, cols)), np.ones((cols, cols))]
                    [other_r, other_p] = [np.ones((cols, cols)), np.ones((cols, cols))]
                    for i in range(self.n_ag):
                        for j in range(self.n_ag):
                            if i == j:
                                r_, p_ = 1., 1.
                                r2_, p2_ = 1., 1.
                                r3_, p3_ = 1., 1.
                            else:
                                r_, p_ = ss.pearsonr(prcc_matrix_school[:, i], prcc_matrix_school[:, j])
                                r2_, p2_ = ss.pearsonr(prcc_matrix_work[:, i], prcc_matrix_work[:, j])
                                r3_, p3_ = ss.pearsonr(prcc_matrix_other[:, i], prcc_matrix_other[:, j])
                            [school_r[j][i], school_p[j][i]] = [r_, p_]
                            [work_r[j][i], work_p[j][i]] = [r2_, p2_]
                            [other_r[j][i], other_p[j][i]] = [r3_, p3_]
                            # print("correlation values age 0 and 1", other_r[0][1])
                            # print("p_values age 0 and other age groups", other_p[0][0:16])
                            # print("p_values age 1 and other age groups", other_p[1][0:16])

                    # Now aggregate the prcc values
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
                    self.prcc_matrix_school = prcc_matrix_school
                    self.prcc_matrix_work = prcc_matrix_work
                    self.prcc_matrix_other = prcc_matrix_other

                elif 'lockdown' in filename_without_ext:
                    prcc_mtx = get_rectangular_matrix_from_upper_triu(prcc_list[:self.upp_tri_size], self.n_ag)
                    self.prcc_mtx = prcc_mtx
                    p_icr = (1 - self.params['p']) * self.params['h'] * self.params['xi']
                    self.p_icr = p_icr

                # PRCC analysis for ICU maximum
                simulation_2 = np.append(sim_data, saved_simulation[:, -self.n_ag - 2].reshape((-1, 1)), axis=1)
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
                    self.prcc_matrix_school_2 = prcc_matrix_school_2
                    self.prcc_matrix_work_2 = prcc_matrix_work_2
                    self.prcc_matrix_other_2 = prcc_matrix_other_2

                elif 'lockdown' in filename_without_ext:
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

    def calculate_p_values(self):
        """
        Calculate the p-value using one sample t-test i.e. for a single contact matrix i.e. school.
        Return: one p-value for each age group and each prcc value
        """
        t_statistic_s, p_value_s = ss.ttest_1samp(self.prcc_matrix_school, popmean=0, axis=1)
        t_statistic_w, p_value_w = ss.ttest_1samp(self.prcc_matrix_work, popmean=0, axis=1)
        t_statistic_o, p_value_o = ss.ttest_1samp(self.prcc_matrix_other, popmean=0, axis=1)

        # aggregate prcc values using the p-values
        p_agg_school, p_agg_work, p_agg_other = [
            np.sum(p_value_s * self.prcc_matrix_school, axis=1),
            np.sum(p_value_w * self.prcc_matrix_work, axis=1),
            np.sum(p_value_o * self.prcc_matrix_other, axis=1)]

        self.prcc_p_values.update(
             {
                 "school": p_agg_school,
                 "work": p_agg_work,
                 "other": p_agg_other
             }
         )

        # PRCC analysis for ICU maximum
        s_statistic, p_s = ss.ttest_1samp(self.prcc_matrix_school_2, popmean=0, axis=1)
        w_statistic, p_w = ss.ttest_1samp(self.prcc_matrix_work_2, popmean=0, axis=1)
        o_statistic, p_o = ss.ttest_1samp(self.prcc_matrix_other_2, popmean=0, axis=1)

        # aggregate prcc values using the p-values
        p_agg_school_2, p_agg_work_2, p_agg_other_2 = [
            np.sum(p_s * self.prcc_matrix_school_2, axis=1),
            np.sum(p_w * self.prcc_matrix_work_2, axis=1),
            np.sum(p_o * self.prcc_matrix_other_2, axis=1)]

        self.prcc_p_values.update(
            {
                "school_2": p_agg_school_2,
                "work_2": p_agg_work_2,
                "other_2": p_agg_other_2
            }
        )

        # get the p-values
        self.p_values.update(
            {
                "school": p_value_s,
                "work": p_value_w,
                "other": p_value_o,
                "school_2": p_s,
                "work_2": p_w,
                "other_2": p_o
            }
        )