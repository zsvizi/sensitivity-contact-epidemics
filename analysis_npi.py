import numpy as np

from plotter import plot_solution_inc, plot_solution_ic
from prcc import get_contact_matrix_from_upper_triu


class AnalysisNPI:
    def __init__(self, sim, susc, base_r0, mtx_type):
        self.sim = sim
        self.susc = susc
        self.base_r0 = base_r0
        self.mtx_type = mtx_type

    def run(self):
        cm_list = []
        legend_list = []
        self.get_full_cm(cm_list, legend_list)
        self.get_reduced_contact(cm_list, legend_list, 3, "school", 0.5)
        self.get_reduced_contact(cm_list, legend_list, 3, "other", 0.5)
        self.get_reduced_contact(cm_list, legend_list, 7, "work", 0.5)
        self.get_reduced_contact(cm_list, legend_list, 8, "work", 0.5)
        self.get_reduced_contact(cm_list, legend_list, 3, "school", 0)
        self.get_reduced_contact(cm_list, legend_list, 3, "other", 0)
        self.get_reduced_contact(cm_list, legend_list, 7, "work", 0)
        self.get_reduced_contact(cm_list, legend_list, 8, "work", 0)
        self.get_reduced_contact(cm_list, legend_list, 5, "work", 0.5)
        self.get_reduced_contact(cm_list, legend_list, 5, "work", 0)

        self.get_fix_reduced_contact(cm_list, legend_list, 3, "school")
        self.get_fix_reduced_contact(cm_list, legend_list, 3, "other")
        self.get_fix_reduced_contact(cm_list, legend_list, 7, "work")
        self.get_fix_reduced_contact(cm_list, legend_list, 8, "work")
        self.get_fix_reduced_contact(cm_list, legend_list, 5, "work")

        t = np.arange(0, 500, 0.5)

        if self.base_r0 == 1.35 and self.susc == 1:
            plot_solution_inc(self.sim, t, self.sim.params,
                              [cm_list[i] for i in [0, 1, 2, 3, 4]], [legend_list[i] for i in [0, 1, 2, 3, 4]],
                              "_R0target_half_".join([str(self.susc), str(self.base_r0)]))
            plot_solution_inc(self.sim, t, self.sim.params,
                              [cm_list[i] for i in [0, 5, 6, 7, 8]], [legend_list[i] for i in [0, 5, 6, 7, 8]],
                              "_R0target_clear_".join([str(self.susc), str(self.base_r0)]))
            plot_solution_inc(self.sim, t, self.sim.params,
                              [cm_list[i] for i in [0, 1, 2, 3, 9]], [legend_list[i] for i in [0, 1, 2, 3, 9]],
                              "_ICUtarget_half_".join([str(self.susc), str(self.base_r0)]))
            plot_solution_inc(self.sim, t, self.sim.params,
                              [cm_list[i] for i in [0, 5, 6, 7, 10]], [legend_list[i] for i in [0, 5, 6, 7, 10]],
                              "_ICUtarget_clear_".join([str(self.susc), str(self.base_r0)]))
            plot_solution_inc(self.sim, t, self.sim.params,
                              [cm_list[i] for i in [0, 11, 12, 13, 14]], [legend_list[i] for i in [0, 11, 12, 13, 14]],
                              "_R0target_fix1M_".join([str(self.susc), str(self.base_r0)]))
            plot_solution_inc(self.sim, t, self.sim.params,
                              [cm_list[i] for i in [0, 11, 12, 13, 15]], [legend_list[i] for i in [0, 11, 12, 13, 15]],
                              "_ICUtarget_fix1M_".join([str(self.susc), str(self.base_r0)]))

            plot_solution_ic(self.sim, t, self.sim.params,
                              [cm_list[i] for i in [0, 1, 2, 3, 4]], [legend_list[i] for i in [0, 1, 2, 3, 4]],
                              "_R0target_half_".join([str(self.susc), str(self.base_r0)]))
            plot_solution_ic(self.sim, t, self.sim.params,
                              [cm_list[i] for i in [0, 5, 6, 7, 8]], [legend_list[i] for i in [0, 5, 6, 7, 8]],
                              "_R0target_clear_".join([str(self.susc), str(self.base_r0)]))
            plot_solution_ic(self.sim, t, self.sim.params,
                              [cm_list[i] for i in [0, 1, 2, 3, 9]], [legend_list[i] for i in [0, 1, 2, 3, 9]],
                              "_ICUtarget_half_".join([str(self.susc), str(self.base_r0)]))
            plot_solution_ic(self.sim, t, self.sim.params,
                              [cm_list[i] for i in [0, 5, 6, 7, 10]], [legend_list[i] for i in [0, 5, 6, 7, 10]],
                              "_ICUtarget_clear_".join([str(self.susc), str(self.base_r0)]))
            plot_solution_ic(self.sim, t, self.sim.params,
                              [cm_list[i] for i in [0, 11, 12, 13, 14]], [legend_list[i] for i in [0, 11, 12, 13, 14]],
                              "_R0target_fix1M_".join([str(self.susc), str(self.base_r0)]))
            plot_solution_ic(self.sim, t, self.sim.params,
                              [cm_list[i] for i in [0, 11, 12, 13, 15]], [legend_list[i] for i in [0, 11, 12, 13, 15]],
                              "_ICUtarget_fix1M_".join([str(self.susc), str(self.base_r0)]))

    def get_full_cm(self, cm_list, legend_list):
        cm = self.sim.contact_matrix
        cm_list.append(cm)
        legend_list.append("Total contact")

    def get_reduced_contact(self, cm_list, legend_list, age_group, contact_type, ratio):
        contact_school = np.copy(self.sim.data.contact_data[contact_type]) * self.sim.age_vector

        contact_school[age_group, :] *= ratio
        contact_school[:, age_group] = contact_school[age_group, :].T
        # contact_school[:, age_group] *= ratio
        # if ratio != 0:
        #     contact_school[age_group, age_group] *= (1/ratio)

        full_contact_matrix = self.sim.contact_matrix - self.sim.data.contact_data[contact_type] + \
            (contact_school / self.sim.age_vector)

        trcm = (full_contact_matrix * self.sim.age_vector)[self.sim.upper_tri_indexes]

        # osszes_elvett = np.sum((self.sim.contact_matrix * self.sim.age_vector)[self.sim.upper_tri_indexes] - trcm)
        # leosztott = np.sum((self.sim.contact_matrix * self.sim.age_vector)[self.sim.upper_tri_indexes] - trcm) / self.sim.age_vector[age_group]
        #
        # print(str(osszes_elvett) + "  " + str(leosztott))

        cm = get_contact_matrix_from_upper_triu(rvector=trcm, age_vector=self.sim.age_vector.reshape(-1, ))

        cm_list.append(cm)
        legend_list.append("{r}% {c_type} reduction of a.g. {ag}".format(r=int((1-ratio)*100),
                                                                         c_type=contact_type, ag=age_group))

    def get_fix_reduced_contact(self, cm_list, legend_list, age_group, contact_type):
        contact_school = np.copy(self.sim.data.contact_data[contact_type]) * self.sim.age_vector

        summa = np.sum(contact_school[age_group, :])

        contact_school[age_group, :] -= 1000000 * contact_school[age_group, :] / summa
        contact_school[:, age_group] = contact_school[age_group, :].T

        full_contact_matrix = self.sim.contact_matrix - self.sim.data.contact_data[contact_type] + \
            (contact_school / self.sim.age_vector)

        trcm = (full_contact_matrix * self.sim.age_vector)[self.sim.upper_tri_indexes]

        # osszes_elvett = np.sum((self.sim.contact_matrix * self.sim.age_vector)[self.sim.upper_tri_indexes] - trcm)
        # leosztott = np.sum((self.sim.contact_matrix * self.sim.age_vector)[self.sim.upper_tri_indexes] - trcm) / self.sim.age_vector[age_group]
        #
        # print(str(osszes_elvett) + "  " + str(leosztott))

        cm = get_contact_matrix_from_upper_triu(rvector=trcm, age_vector=self.sim.age_vector.reshape(-1, ))

        cm_list.append(cm)
        legend_list.append("1M {c_type} reduction of a.g. {ag}".format(c_type=contact_type, ag=age_group))
