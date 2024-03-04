import numpy as np
from src.plotter import Plotter


class ContactManipulation:
    def __init__(self, sim_obj, contact_matrix: np.ndarray,
                 contact_home: np.ndarray,
                 susc: float, base_r0: float, params: str, model: str, data):
        self.sim_obj = sim_obj

        self.contact_matrix = contact_matrix
        self.contact_home = contact_home
        self.base_r0 = base_r0
        self.susc = susc
        self.params = params
        self.model = model
        self.plotter = Plotter(sim_obj=sim_obj, data=data)

    def run_plots(self, model: str):
        cm_list_orig = []
        legend_list_orig = []

        self.get_full_cm(cm_list_orig, legend_list_orig)

        if model == "rost":
            t = np.arange(0, 1200, 0.5)
        elif model == "chikina":
            t = np.arange(0, 2500, 0.5)
        else:
            t = np.arange(0, 400, 0.5)
        ratios = [0.5, 0.75]
        for ratio in ratios:
            cm_list = cm_list_orig.copy()
            legend_list = legend_list_orig.copy()
            for i in range(self.sim_obj.n_ag):
                self.generate_contact_matrix(cm_list, legend_list, i, ratio)
            # epidemic size
            self.plotter.plot_epidemic_size(
                time=t, cm_list=cm_list, legend_list=legend_list,
                model=self.model, ratio=ratio)
            # epidemic peak
            self.plotter.plot_peak_size_epidemic(
                time=t, params=self.params, cm_list=cm_list, legend_list=legend_list,
                title_part="_epidemic_size_".join([str(self.susc), str(self.base_r0)]),
                model=self.model, ratio=ratio)
            # plot icu size
            self.plotter.plot_icu_size(
                time=t, params=self.params,
                cm_list=cm_list, susc=self.susc, base_r0=self.base_r0,
                legend_list=legend_list,
                title_part="_icu_".join([str(self.susc), str(self.base_r0), str(ratio)]),
                model=self.model, ratio=ratio)
            # number of hospitalized
            self.plotter.plot_solution_hospitalized_size(
                time=t, params=self.params, cm_list=cm_list,
                legend_list=legend_list,
                title_part="_hospitalized_peak_".join([str(self.susc), str(self.base_r0)]),
                model=self.model, ratio=ratio)
            # final death size
            self.plotter.plot_solution_final_death_size(
                time=t, params=self.params, cm_list=cm_list,
                legend_list=legend_list,
                title_part="_final_deaths_".join([str(self.susc), str(self.base_r0)]),
                model=self.model, ratio=ratio)

    def get_full_cm(self, cm_list, legend_list):
        cm = self.contact_matrix
        cm_list.append(cm)
        legend_list.append("Total contact")

    def generate_contact_matrix(self, cm_list, legend_list, age_group, ratio):
        if self.contact_matrix is None:
            contact_matrix_spec = self._calculate_contact_matrix_spec(age_group, ratio)
            full_contact_matrix = self.contact_home + contact_matrix_spec
            return full_contact_matrix
        else:
            contact_matrix_spec = np.copy(self.contact_matrix)
            self._update_contact_matrix_spec(contact_matrix_spec, age_group, ratio)
            cm_list.append(contact_matrix_spec)
            legend_list.append("{r}% reduction of a.g. {ag}".format(r=int((1 - ratio) * 100),
                                                                    ag=age_group))

    def _calculate_contact_matrix_spec(self, age_group, ratio):
        contact_matrix_spec = np.copy(self.contact_matrix) - np.copy(self.contact_home)
        self._apply_ratio_to_contact_matrix(contact_matrix_spec, age_group, ratio)
        return contact_matrix_spec

    def _update_contact_matrix_spec(self, contact_matrix_spec, age_group, ratio):
        return self._apply_ratio_to_contact_matrix(contact_matrix_spec, age_group, ratio)

    @staticmethod
    def _apply_ratio_to_contact_matrix(contact_matrix_spec, age_group, ratio):
        contact_matrix_spec[age_group, :] *= ratio
        contact_matrix_spec[:, age_group] *= ratio
        contact_matrix_spec[age_group, age_group] *= (1 / ratio if ratio > 0.0 else 0.0)
