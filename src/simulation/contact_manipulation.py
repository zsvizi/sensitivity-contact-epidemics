import numpy as np

from src.plotter import Plotter


class ContactManipulation:
    def __init__(self, sim_obj, susc: float, base_r0: float, model: str):
        self.sim_obj = sim_obj
        self.base_r0 = base_r0
        self.susc = susc
        self.model = model

        self.contact_matrix = sim_obj.contact_matrix
        self.contact_home = sim_obj.contact_home
        self.params = sim_obj.params

        self.plotter = Plotter(sim_obj=sim_obj, data=sim_obj.data)

    def run_plots(self):
        cm_list_orig = []
        legend_list_orig = []
        self.get_full_cm(cm_list_orig, legend_list_orig)

        if self.model == "rost":
            t = np.arange(0, 1200, 0.5)
        elif self.model == "chikina":
            t = np.arange(0, 2500, 0.5)
        else:
            t = np.arange(0, 500, 0.5)

        ratios = [0.75, 0.5]
        for ratio in ratios:
            cm_list = cm_list_orig.copy()
            legend_list = legend_list_orig.copy()
            for i in range(self.sim_obj.n_ag):
                self.generate_contact_matrix(
                    cm_list=cm_list, legend_list=legend_list,
                    age_group=i, ratio=ratio)

            # epidemic size and epidemic peak
            self.plotter.plot_epidemic_peak_and_size(
                time=t, cm_list=cm_list, legend_list=legend_list,
                model=self.model, ratio=ratio)
            # plot icu size
            if self.model in ["rost", "chikina", "moghadas"]:
                self.plotter.plot_icu_size(
                    time=t, cm_list=cm_list, legend_list=legend_list,
                    model=self.model, ratio=ratio)
                # number of hospitalized
                self.plotter.plot_solution_hospitalized_size(
                    time=t, cm_list=cm_list, legend_list=legend_list,
                    model=self.model, ratio=ratio)
                # final death size
                self.plotter.plot_solution_final_death_size(
                     time=t, cm_list=cm_list, legend_list=legend_list,
                     model=self.model, ratio=ratio)

    def get_full_cm(self, cm_list, legend_list):
        cm = self.contact_matrix
        cm_list.append(cm)
        legend_list.append("Total contact")

    def generate_contact_matrix(self, cm_list, legend_list, age_group, ratio):
        if self.contact_matrix is None:
            contact_matrix_spec = self._calculate_contact_matrix_spec(
                age_group=age_group, ratio=ratio)
            full_contact_matrix = self.contact_home + contact_matrix_spec
            return full_contact_matrix
        else:
            contact_matrix_spec = np.copy(self.contact_matrix)
            self._apply_ratio_to_contact_matrix(
                contact_matrix_spec=contact_matrix_spec,
                age_group=age_group, ratio=ratio)
            cm_list.append(contact_matrix_spec)
            legend_list.append("{r}% reduction of a.g. {ag}".format(r=int((1 - ratio) * 100),
                                                                    ag=age_group))

    def _calculate_contact_matrix_spec(self, age_group, ratio):
        contact_matrix_spec = np.copy(self.contact_matrix) - np.copy(self.contact_home)
        self._apply_ratio_to_contact_matrix(
            contact_matrix_spec=contact_matrix_spec, age_group=age_group, ratio=ratio)
        return contact_matrix_spec

    @staticmethod
    def _apply_ratio_to_contact_matrix(contact_matrix_spec, age_group, ratio):
        contact_matrix_spec[age_group, :] *= ratio
        contact_matrix_spec[:, age_group] *= ratio
        contact_matrix_spec[age_group, age_group] *= (1 / ratio if ratio > 0.0 else 0.0)
