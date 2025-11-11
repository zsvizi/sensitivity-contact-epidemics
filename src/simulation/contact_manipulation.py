import numpy as np

import src
from src.plotter import Plotter


class ContactManipulation:
    """
    A class responsible for analyzing and visualizing the effects of contact matrix manipulation

    It modifies the contact matrices by applying reduction ratios to specific age groups
    and generates plots for epidemic outcomes such as epidemic size, ICU demand,
    hospitalization, and death size.
    """

    def __init__(self, sim_obj: src.SimulationNPI, susc: float, base_r0: float, model: str):
        """
        Initializes the class with a simulation reference and key parameters.

        :param sim_obj: Simulation object that provides model data, parameters, and matrices.
        :param float susc: Susceptibility scaling factor applied to the model.
        :param float base_r0: Baseline reproduction number (R_0) used for scaling the infection rate.
        :param str model: Name of the Epidemiological model (e.g., "rost", "chikina", "moghadas").
        """
        self.sim_obj = sim_obj
        self.base_r0 = base_r0
        self.susc = susc
        self.model = model

        self.contact_matrix = sim_obj.contact_matrix
        self.contact_home = sim_obj.contact_home
        self.params = sim_obj.params

        self.plotter = Plotter(sim_obj=sim_obj, data=sim_obj.data)

    def run_plots(self) -> None:
        """
        Executes all contact manipulation analyses and generate the corresponding plots.

        The method iterates through predefined contact reduction ratios and age groups,
        creates modified contact matrices, and visualizes various epidemiological metrics
        (epidemic peak, size, ICU, hospitalization, and death).
        """
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

            # Epidemic size and peak
            self.plotter.plot_epidemic_peak_and_size(
                time=t, cm_list=cm_list, legend_list=legend_list,
                model=self.model, ratio=ratio)

            # ICU, hospitalization, and death outcomes (if model supports them)
            if self.model in ["rost", "chikina", "moghadas"]:
                self.plotter.plot_icu_size(
                    time=t, cm_list=cm_list, legend_list=legend_list,
                    model=self.model, ratio=ratio)

                self.plotter.plot_solution_hospitalized_size(
                    time=t, cm_list=cm_list, legend_list=legend_list,
                    model=self.model, ratio=ratio)

                self.plotter.plot_solution_final_death_size(
                    time=t, cm_list=cm_list, legend_list=legend_list,
                    model=self.model, ratio=ratio)

    def get_full_cm(self, cm_list: list, legend_list: list) -> None:
        """
        Appends the full contact matrix and its label to the provided lists.

        :param list cm_list: List of contact matrices to which the full matrix will be added.
        :param list legend_list: List of legends corresponding to the contact matrices.
        """
        cm = self.contact_matrix
        cm_list.append(cm)
        legend_list.append("Total contact")

    def generate_contact_matrix(self, cm_list: list, legend_list: list, age_group: int, ratio: float) -> np.ndarray:
        """
        Generate a modified contact matrix for a specific age group by applying a reduction ratio.

        :param list cm_list: List to which the modified contact matrix will be appended.
        :param list legend_list: List to which the corresponding legend will be appended.
        :param int age_group: Index of the age group to which the ratio is applied.
        :param float ratio: Reduction ratio
        :return np.ndarray: The modified contact matrix.
        """
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
            legend_list.append(f"{int((1 - ratio) * 100)}% reduction of a.g. {age_group}")

    def _calculate_contact_matrix_spec(self, age_group: int, ratio: float) -> np.ndarray:
        """
        Computes a specific contact matrix excluding home contacts and apply a reduction ratio.

        :param int age_group: Index of the age group affected by the contact reduction.
        :param float ratio: Reduction ratio applied to contacts.
        :return np.ndarray: The modified contact matrix without home contacts.
        """
        contact_matrix_spec = np.copy(self.contact_matrix) - np.copy(self.contact_home)
        self._apply_ratio_to_contact_matrix(
            contact_matrix_spec=contact_matrix_spec, age_group=age_group, ratio=ratio)
        return contact_matrix_spec

    @staticmethod
    def _apply_ratio_to_contact_matrix(contact_matrix_spec: np.ndarray, age_group: int, ratio: float) -> None:
        """
        Applies a reduction ratio to the specified age group's contacts in the contact matrix.

        This function symmetrically scales contacts for the given age group and ensures
        the diagonal element is adjusted to maintain proper normalization.

        :param np.ndarray contact_matrix_spec: The contact matrix to modify.
        :param int age_group: Index of the age group being scaled.
        :param float ratio: Ratio of retained contacts
        """
        contact_matrix_spec[age_group, :] *= ratio
        contact_matrix_spec[:, age_group] *= ratio
        contact_matrix_spec[age_group, age_group] *= (1 / ratio if ratio > 0.0 else 0.0)
