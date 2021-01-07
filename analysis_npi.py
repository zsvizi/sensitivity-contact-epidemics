import numpy as np

from plotter import plot_solution_inc
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
        self.get_half_contact(cm_list, legend_list, 3, "school")
        self.get_half_contact(cm_list, legend_list, 3, "other")
        self.get_half_contact(cm_list, legend_list, 7, "work")
        self.get_half_contact(cm_list, legend_list, 8, "work")

        t = np.arange(0, 500, 0.5)

        if self.base_r0 == 1.35 and self.susc == 1:
            plot_solution_inc(self.sim, t, self.sim.params, cm_list, legend_list,
                              "_P1_".join([str(self.susc), str(self.base_r0)]))

    def get_full_cm(self, cm_list, legend_list):
        cm = self.sim.contact_matrix
        cm_list.append(cm)
        legend_list.append("Total contact")

    def get_school_reduction(self, cm_list, legend_list):
        contact_work = np.copy(self.sim.data.contact_data["work"]) * self.sim.age_vector
        fixed_number = contact_work[7, 8] * 0.5
        contact_work[7, 8] -= fixed_number
        full_contact_matrix = self.sim.data.contact_data["home"] + self.sim.data.contact_data["school"] + \
            self.sim.data.contact_data["other"]
        cm_school_reduction = \
            (full_contact_matrix * self.sim.age_vector + contact_work)[self.sim.upper_tri_indexes]

        cm = get_contact_matrix_from_upper_triu(rvector=cm_school_reduction,
                                                age_vector=self.sim.age_vector.reshape(-1, ))
        cm_list.append(cm)
        legend_list.append("50% School reduction of a.g. 3")

    def get_school_reduction_fix(self, cm_list, legend_list):
        contact_work = np.copy(self.sim.data.contact_data["work"]) * self.sim.age_vector
        fixed_number = contact_work[7, 8] * 0.5
        contact_school = np.copy(self.sim.data.contact_data["school"]) * self.sim.age_vector
        contact_school[3, 3] -= fixed_number
        full_contact_matrix = self.sim.data.contact_data["home"] + self.sim.data.contact_data["work"] + \
            self.sim.data.contact_data["other"]
        cm_fixed_reduced_school = \
            (full_contact_matrix * self.sim.age_vector + contact_school)[self.sim.upper_tri_indexes]

        cm = get_contact_matrix_from_upper_triu(rvector=cm_fixed_reduced_school,
                                                age_vector=self.sim.age_vector.reshape(-1, ))
        cm_list.append(cm)
        legend_list.append("Reduction with fixed number")

    def get_half_contact(self, cm_list, legend_list, age_group, contact_type):
        contact_school = np.copy(self.sim.data.contact_data[contact_type]) * self.sim.age_vector

        contact_school[age_group, :] *= 0.5
        contact_school[:, age_group] *= 0.5
        contact_school[age_group, age_group] *= 2

        full_contact_matrix = self.sim.contact_matrix - self.sim.data.contact_data[contact_type] + \
            (contact_school / self.sim.age_vector)

        trcm = (full_contact_matrix * self.sim.age_vector)[self.sim.upper_tri_indexes]

        print(np.sum((self.sim.contact_matrix * self.sim.age_vector)[self.sim.upper_tri_indexes] - trcm) / self.sim.age_vector[age_group])

        cm = get_contact_matrix_from_upper_triu(rvector=trcm, age_vector=self.sim.age_vector.reshape(-1, ))

        cm_list.append(cm)
        legend_list.append("50% {c_type} reduction of a.g. {ag}".format(c_type=contact_type, ag=age_group))