import numpy as np

from src.plotter import Plotter


class ContactManipulation:
    def __init__(self, sim_obj, contact_matrix, contact_home,
                 susc, base_r0, params):
        self.sim_obj = sim_obj

        self.contact_matrix = contact_matrix
        self.contact_home = contact_home
        self.base_r0 = base_r0
        self.susc = susc
        self.params = params

    def run_plots(self):

        cm_list = []
        legend_list = []

        self.get_full_cm(cm_list, legend_list)
        t = np.arange(0, 500, 0.5)

        for i in range(16):
            self.get_reduced_contact(cm_list, legend_list, i, 0.5)

        Plotter.plot_solution_hospitalized_peak_size(
            self.sim_obj, time=t,
            params=self.params, cm_list=cm_list,
            legend_list=legend_list,
            title_part="_hospitalized_peak_".join([str(self.susc),
                                                 str(self.base_r0)]))

        Plotter.plot_solution_final_death_size(self.sim_obj, time=t,
                                               params=self.params, cm_list=cm_list,
                                               legend_list=legend_list,
                                               title_part="_final_deaths_".join([str(self.susc),
                                                                                   str(self.base_r0)]))

    def get_full_cm(self, cm_list, legend_list):
        cm = self.contact_matrix
        cm_list.append(cm)
        legend_list.append("Total contact")

    def get_reduced_contact(self, cm_list, legend_list, age_group, ratio):
        if self.contact_matrix is None:
            contact_matrix_spec = np.copy(self.contact_matrix) - \
                                  np.copy(self.contact_home)

            contact_matrix_spec[age_group, :] *= ratio
            contact_matrix_spec[:, age_group] *= ratio
            contact_matrix_spec[age_group, age_group] *= (1 / ratio if ratio > 0.0 else 0.0)

            full_contact_matrix = self.contact_home + contact_matrix_spec
            return full_contact_matrix
        else:
            contact_matrix_spec = np.copy(self.contact_matrix)

            contact_matrix_spec[age_group, :] *= ratio
            contact_matrix_spec[:, age_group] *= ratio
            contact_matrix_spec[age_group, age_group] *= (1 / ratio if ratio > 0.0 else 0.0)

        cm_list.append(contact_matrix_spec)
        legend_list.append("{r}% reduction of a.g. {ag}".format(r=int((1 - ratio) * 100),
                                                                ag=age_group))

