import numpy as np

from plotter import plot_solution_inc
from prcc import get_contact_matrix_from_upper_triu


class Analysis:
    def __init__(self, sim, susc, base_r0, mtx_type):
        self.sim = sim
        self.susc = susc
        self.base_r0 = base_r0
        self.mtx_type = mtx_type

    def run(self):
        cm_list = []
        legend_list = []
        self.get_full_cm(cm_list, legend_list)
        self.get_school_reduction(cm_list, legend_list)
        self.get_school_reduction_fix(cm_list, legend_list)

        t = np.arange(0, 500, 0.5)
        plot_solution_inc(self.sim, t, self.sim.params, cm_list, legend_list,
                          "_P1_".join([str(self.susc), str(self.base_r0)]))

    def get_full_cm(self, cm_list, legend_list):
        cm_full = self.sim.upper_limit_matrix[self.sim.upper_tri_indexes]
        print(np.sum(cm_full))
        cm = get_contact_matrix_from_upper_triu(rvector=cm_full,
                                                age_vector=self.sim.age_vector.reshape(-1, ))
        cm_list.append(cm)
        legend_list.append("Total contact")

    def get_school_reduction(self, cm_list, legend_list):
        contact_work = np.copy(self.sim.data.contact_data["work"]) * self.sim.age_vector
        fixed_number = contact_work[7, 8] * 0.5
        print(fixed_number)
        contact_work[7, 8] -= fixed_number
        full_contact_matrix = self.sim.data.contact_data["home"] + self.sim.data.contact_data["school"] + \
            self.sim.data.contact_data["other"]
        cm_school_reduction = \
            (full_contact_matrix * self.sim.age_vector + contact_work)[self.sim.upper_tri_indexes]
        print(np.sum(cm_school_reduction))

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
        print(np.sum(cm_fixed_reduced_school))

        cm = get_contact_matrix_from_upper_triu(rvector=cm_fixed_reduced_school,
                                                age_vector=self.sim.age_vector.reshape(-1, ))
        cm_list.append(cm)
        legend_list.append("Reduction with fixed number")
