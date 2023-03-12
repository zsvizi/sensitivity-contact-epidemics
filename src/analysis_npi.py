import numpy as np

from src.plotter import Plotter


class AnalysisNPI:
    def __init__(self, sim, susc, base_r0, mtx_type=None, kappa=None) -> None:
        self.sim = sim
        self.susc = susc
        self.base_r0 = base_r0
        self.mtx_type = mtx_type
        self.kappa = kappa

    def run(self):
        cm_list = []
        legend_list = []

        self.get_full_cm(cm_list, legend_list)
        for i in range(16):
            self.get_reduced_contact(cm_list, legend_list, i, None, max(0.5, self.kappa))

        t = np.arange(0, 350, 0.5)

        Plotter.plot_solution_inc(self.sim, t, self.sim.params, cm_list, legend_list,
                                  "_R0target_reduce_".join([str(self.susc), str(self.base_r0)]))

        # R0 = 1.35, Susc = 1, Target: ICU
        Plotter.plot_solution_ic(self.sim, t, self.sim.params, cm_list, legend_list,
                                 "_ICUtarget_reduce_".join([str(self.susc), str(self.base_r0)]))

    def get_full_cm(self, cm_list, legend_list):
        cm = self.sim.contact_matrix
        cm_list.append(cm)
        legend_list.append("Total contact")

    def get_reduced_contact(self, cm_list, legend_list, age_group, contact_type, ratio):
        if contact_type is None:
            contact_matrix_spec = np.copy(self.sim.contact_matrix) - np.copy(self.sim.data.contact_data["home"])

            contact_matrix_spec[age_group, :] *= ratio
            contact_matrix_spec[:, age_group] *= ratio
            contact_matrix_spec[age_group, age_group] *= (1/ratio if ratio > 0.0 else 0.0)

            full_contact_matrix = self.sim.data.contact_data["home"] + contact_matrix_spec

        else:
            contact_matrix_spec = np.copy(self.sim.data.contact_data[contact_type])

            contact_matrix_spec[age_group, :] *= ratio
            contact_matrix_spec[:, age_group] *= ratio
            contact_matrix_spec[age_group, age_group] *= (1/ratio if ratio > 0.0 else 0.0)

            full_contact_matrix = self.sim.contact_matrix - self.sim.data.contact_data[contact_type] + \
                contact_matrix_spec

        cm_list.append(full_contact_matrix)
        legend_list.append("{r}% reduction of a.g. {ag}".format(r=int((1-ratio)*100), ag=age_group))

    def get_reduced_contact_one_element(self, cm_list, legend_list, element, ratio):
        contact_matrix_spec = np.copy(self.sim.contact_matrix) - np.copy(self.sim.data.contact_data["home"])
        contact_matrix_spec[element[0], element[1]] *= ratio
        full_contact_matrix = self. sim.data.contact_data["home"] + contact_matrix_spec

        cm_list.append(full_contact_matrix)
        legend_list.append("{r}% reduction of element {ag}".format(r=int((1 - ratio) * 100), ag=element))

    def get_fix_reduced_contact(self, cm_list, legend_list, age_group, contact_type):
        cm_spec_total = np.copy(self.sim.data.contact_data[contact_type]) * self.sim.age_vector

        all_spec_contacts = np.sum(cm_spec_total[age_group, :])

        cm_spec_total[age_group, :] -= 1000000 * cm_spec_total[age_group, :] / all_spec_contacts
        cm_spec_total[:, age_group] = cm_spec_total[age_group, :].T
        contact_matrix_spec = cm_spec_total / self.sim.age_vector

        full_contact_matrix = self.sim.contact_matrix - self.sim.data.contact_data[contact_type] + contact_matrix_spec

        cm_list.append(full_contact_matrix)
        legend_list.append("1M {c_type} reduction of a.g. {ag}".format(c_type=contact_type, ag=age_group))
