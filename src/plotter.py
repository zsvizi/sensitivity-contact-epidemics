import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import LogLocator, LogFormatterSciNotation as LogFormatter
import matplotlib.colors as colors
from matplotlib.tri import Triangulation
import numpy as np
import pandas as pd
import seaborn as sns

from src.dataloader import DataLoader
from src.simulation_base import SimulationBase
from src.prcc import get_rectangular_matrix_from_upper_triu

plt.style.use('seaborn-whitegrid')
matplotlib.use('agg')


class Plotter:
    def __init__(self, data: DataLoader, sim_obj: SimulationBase) -> None:

        self.deaths = None
        self.hospitalized = None

        self.data = data
        self.sim_obj = sim_obj
        self.contact_full = np.array([])

    def plot_contact_matrices_models(self, filename, model, contact_data,
                                     plot_total_contact: bool = True):
        """
        Plot contact matrices for different models and save it in sub-directory
        :param filename: The filename prefix for the saved PDF files.
        :param model: The model name representing the type of contact matrices.
        :param contact_data: A dictionary containing contact matrices for different
        categories. Keys represent contact categories and values represent the contact matrices.
        :param plot_total_contact: for plotting total contacts
        :return: Heatmaps for different models
        """
        output_dir = f"sens_data/contact_matrices/{model}"  # Subdirectory for each model
        os.makedirs(output_dir, exist_ok=True)

        cmaps = {
            "rost": {"Home": "inferno", "Work": "inferno", "School": "inferno", "Other": "inferno", "Full": "inferno"},
            "chikina": {"Home": "inferno", "Work": "inferno", "School": "inferno", "Other": "inferno",
                        "Full": "inferno"},
            "moghadas": {"Home": "inferno", "All": "inferno"},
            "seir": {"Physical": "inferno", "All": "inferno"}
        }

        labels_dict = {
            "rost": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                     "55-59", "60-64", "65-69", "70-74", "75+"],
            "chikina": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                        "55-59", "60-64", "65-69", "70-74", "75-79", "80+"],
            "moghadas": ["0-19", "20-49", "50-65", "65+"],
            "seir": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                     "55-59", "60-64", "65-69", "70+"]
        }

        labels = labels_dict.get(model, [])
        if not labels:
            raise ValueError("Invalid model provided")

        contact_full = np.array([contact_data[i] for i in cmaps[model].keys() if i != "Full"]).sum(axis=0)

        for contact_type, cmap in cmaps[model].items():
            contacts = contact_data[contact_type] if contact_type != "Full" else contact_full
            contact_matrix = pd.DataFrame(contacts, columns=range(self.sim_obj.n_ag),
                                          index=range(self.sim_obj.n_ag))

            # Turn off axis
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Create heatmap
            if model in ["rost", "chikina"]:
                if plot_total_contact:
                    sns.heatmap(contact_matrix * self.data.age_data, cmap=cmap, square=True,
                                cbar=contact_type == "Full", ax=ax)
                else:
                    sns.heatmap(contact_matrix, cmap=cmap, vmin=0, vmax=10, square=True,
                                cbar=contact_type == "Full", ax=ax)
            else:
                if plot_total_contact:
                    sns.heatmap(contact_matrix * self.data.age_data, cmap=cmap, square=True,
                                cbar=contact_type == "Full", ax=ax)
                else:
                    sns.heatmap(contact_matrix, cmap=cmap, vmin=0, vmax=10, square=True,
                                cbar=contact_type == "All", ax=ax)
            # Rotate y tick labels
            plt.yticks(rotation=0)
            ax.invert_yaxis()

            plt.title(f"{contact_type} contact", fontsize=25, fontweight="bold")

            plt.savefig(os.path.join(output_dir, f"{filename}_{contact_type}.pdf"), format="pdf", bbox_inches='tight')
            plt.close()

    def construct_triangle_grids_prcc_p_value(self):
        """
        construct_triangle_grids_prcc_p_value(prcc_vector, p_values)
        :return: A list containing triangulation objects for PRCC and p-values.
        """
        # vertices of the little squares
        xv, yv = np.meshgrid(np.arange(-0.5, self.sim_obj.n_ag), np.arange(-0.5,
                                                                           self.sim_obj.n_ag))
        # centers of the little square
        xc, yc = np.meshgrid(np.arange(0, self.sim_obj.n_ag), np.arange(0, self.sim_obj.n_ag))
        x = np.concatenate([xv.ravel(), xc.ravel()])
        y = np.concatenate([yv.ravel(), yc.ravel()])
        triangles_prcc = [(i + j * (self.sim_obj.n_ag + 1), i + 1 + j * (self.sim_obj.n_ag + 1),
                           i + (j + 1) * (self.sim_obj.n_ag + 1))
                          for j in range(self.sim_obj.n_ag) for i in range(self.sim_obj.n_ag)]
        triangles_p = [(i + 1 + j * (self.sim_obj.n_ag + 1), i + 1 + (j + 1) *
                        (self.sim_obj.n_ag + 1),
                        i + (j + 1) * (self.sim_obj.n_ag + 1))
                       for j in range(self.sim_obj.n_ag) for i in range(self.sim_obj.n_ag)]
        triang = [Triangulation(x, y, triangles, mask=None)
                  for triangles in [triangles_prcc, triangles_p]]
        return triang

    def get_mask_and_values(self, prcc_vector, p_values):
        prcc_mtx = get_rectangular_matrix_from_upper_triu(
            rvector=prcc_vector[:self.sim_obj.upper_tri_size],
            matrix_size=self.sim_obj.n_ag)
        p_values_mtx = get_rectangular_matrix_from_upper_triu(
            rvector=p_values[:self.sim_obj.upper_tri_size],
            matrix_size=self.sim_obj.n_ag)
        values_all = [prcc_mtx, p_values_mtx]
        values = np.triu(values_all, k=0)

        # get the masked values
        mask = np.where(values[0] == 0, np.nan, values_all)
        return mask

    def plot_prcc_p_values_as_heatmap(self, prcc_vector,
                                      p_values, filename_to_save, plot_title, option):
        """
        Prepares for plotting PRCC and p-values as a heatmap.
        :param prcc_vector: (numpy.ndarray): The PRCC vector.
        :param p_values: (numpy.ndarray): The p-values vector.
        :param filename_to_save: : The filename to save the plot.
        :param plot_title: The title of the plot.
        :param option: (str): target options for epidemic size.
        :return: None
        """
        p_value_cmap = ListedColormap(['Orange', 'red', 'darkred'])
        # p_valu = ListedColormap(['black', 'black', 'violet', 'violet', 'purple', 'grey',
        #                          'lightgreen',
        #                          'green', 'green', 'darkgreen'])
        cmaps = ["Greens", p_value_cmap]

        log_norm = colors.LogNorm(vmin=1e-3, vmax=1e0)  # used for p_values
        norm = plt.Normalize(vmin=0, vmax=1)  # used for PRCC_values
        # norm = colors.Normalize(vmin=-1, vmax=1e0)  # used for p_values

        fig, ax = plt.subplots()
        triang = self.construct_triangle_grids_prcc_p_value()
        mask = self.get_mask_and_values(prcc_vector=prcc_vector, p_values=p_values)
        images = [ax.tripcolor(t, np.ravel(val), cmap=cmap, ec="white")
                  for t, val, cmap in zip(triang,
                                          mask, cmaps)]

        # fig.colorbar(images[0], ax=ax, shrink=0.7, aspect=20 * 0.7)  # for the prcc values
        cbar = fig.colorbar(images[0], ax=ax, shrink=0.7, aspect=20 * 0.7, pad=0.1)

        cbar_pval = fig.colorbar(images[1], ax=ax, shrink=0.7, aspect=20 * 0.7, pad=0.1)

        images[1].set_norm(norm=log_norm)
        images[0].set_norm(norm=norm)

        locator = LogLocator()
        formatter = LogFormatter()
        cbar_pval.locator = locator
        cbar_pval.formatter = formatter
        cbar_pval.update_normal(images[1])
        cbar.update_normal(images[0])  # add

        ax.set_xticks(range(self.sim_obj.n_ag))
        ax.set_yticks(range(self.sim_obj.n_ag))
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set(frame_on=False)
        plt.gca().grid(which='minor', color='gray', linestyle='-', linewidth=1)
        ax.margins(x=0, y=0)
        ax.set_aspect('equal', 'box')  # square cells
        plt.tight_layout()
        plt.title(plot_title, y=1.03, fontsize=25)
        plt.title(plot_title, y=1.03, fontsize=25)
        plt.close()

    def generate_prcc_p_values_heatmaps(self, prcc_vector,
                                        p_values, filename_without_ext, option):
        """
        Generates actual PRCC and p-values masked heatmaps.
        :param prcc_vector: (numpy.ndarray): The PRCC vector.
        :param p_values: (numpy.ndarray): The p-values vector.
        :param filename_without_ext: (str): The filename prefix for the saved plot.
        :param option: (str): target options for epidemic size.
        :return: masked Heatmaps
        """
        if option:
            os.makedirs(os.path.join("sens_data", option, "prcc_plot"), exist_ok=True)
            save_path = os.path.join("sens_data", option, "prcc_plot",
                                     filename_without_ext + '.pdf')
        else:
            os.makedirs("sens_data/prcc_plot", exist_ok=True)
            save_path = os.path.join("sens_data", "prcc_plot", filename_without_ext + '.pdf')

        # os.makedirs("sens_data/heatmap", exist_ok=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.margins(0, tight=False)
        title_list = filename_without_ext.split("_")
        plot_title = '$\\overline{\\mathcal{R}}_0=$' + title_list[1]

        self.plot_prcc_p_values_as_heatmap(prcc_vector=prcc_vector, p_values=p_values,
                                           filename_to_save="prcc_plot" +
                                                            filename_without_ext + "_R0",
                                           plot_title=plot_title, option=option)

    @staticmethod
    def aggregated_prcc_pvalues_plots(param_list, prcc_vector, std_values,
                                      filename_to_save, plot_title, option):
        """
              Prepares for plotting aggregated PRCC and standard values as error bars.
              :param param_list: list of the parameters
              :param prcc_vector: (numpy.ndarray): The aggregated PRCC vector.
              :param std_values: (numpy.ndarray): The std values from calculating
               aggregated PRCC vector.
              :param filename_to_save: : The filename to save the plots.
              :param option: (str): target options for epidemic size.
              :param plot_title: The title of the plots.
              :return: None
              """
        if option:
            os.makedirs(os.path.join("sens_data", option, "agg_plot"), exist_ok=True)
            save_path = os.path.join("sens_data", option, "agg_plot",
                                     filename_to_save + '.pdf')
        else:
            os.makedirs("sens_data/agg_plot", exist_ok=True)
            save_path = os.path.join("sens_data", "agg_plot", filename_to_save + '.pdf')
        # os.makedirs("sens_data/PRCC_PVAL_PLOT", exist_ok=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.margins(0, tight=False)
        xp = range(param_list)
        plt.figure(figsize=(15, 12))
        plt.tick_params(direction="in")
        # fig, ax = plt.subplots()
        plt.bar(xp, list(abs(prcc_vector)), align='center', width=0.8, alpha=0.5,
                color="g", label="PRCC")
        for pos, y, err in zip(xp, list(abs(prcc_vector)), list(abs(std_values))):
            plt.errorbar(pos, y, err, lw=4, capthick=4, fmt="or",
                         markersize=5, capsize=4, ecolor="r", elinewidth=4)
        plt.xticks(ticks=xp, rotation=90)
        plt.yticks(ticks=np.arange(-1, 1.0, 0.2))
        plt.legend([r'$\mathrm{\textbf{P}}$', r'$\mathrm{\textbf{s}}$'])
        axes = plt.gca()
        axes.set_ylim([0, 1.0])
        plt.xlabel('age groups', labelpad=10, fontsize=20)
        plt.title(plot_title, y=1.03, fontsize=20)
        plt.savefig(save_path, format="pdf", bbox_inches='tight')
        plt.close()

    def plot_aggregation_prcc_pvalues(self, prcc_vector, std_values,
                                      filename_without_ext, option):
        """
        Generates actual aggregated PRCC plots with std values as error bars.
        :param prcc_vector: (numpy.ndarray): The PRCC vector.
        :param std_values: (numpy.ndarray): standard deviation values.
        :param filename_without_ext: (str): The filename prefix for the saved plot.
        :param option: (str): target options for epidemic size.
        :return: bar plots with error bars
        """
        title_list = filename_without_ext.split("_")
        plot_title = '$\\overline{\\mathcal{R}}_0=$' + title_list[1]
        self.aggregated_prcc_pvalues_plots(param_list=self.sim_obj.n_ag,
                                           prcc_vector=prcc_vector, std_values=std_values,
                                           filename_to_save=filename_without_ext,
                                           plot_title=plot_title, option=option)

    @staticmethod
    def plot_model_max_values(max_values, model: str,
                              plot_title: str = "Y"):
        """
        This method Loads the max values from different targets and iterates over
        each directory in directory_column_orders. It loads the corresponding CSV files,
        creates DataFrames, sets column names based on the specified column orders,
        concatenates them, and plots the heatmap for each directory.
        Then, it saves the heatmap under each directory as a pdf file.
        :param max_values: dataframe of max values for each age group
        :param model: rost, moghadas, chikina, seir
        :param plot_title: title of the heatmap
        :return: Heatmaps
        """
        directory_column_orders = [
            (f"./sens_data/Epidemic/Epidemic_values", [
                "Epidemic_values/0.5_1.2_ratio_0.25", "Epidemic_values/0.5_1.2_ratio_0.5",
                "Epidemic_values/0.5_1.8_ratio_0.25", "Epidemic_values/0.5_1.8_ratio_0.5",
                "Epidemic_values/0.5_1.8_ratio_0.25", "Epidemic_values/0.5_1.8_ratio_0.5",
                "Epidemic_values/0.5_1.8_ratio_0.25", "Epidemic_values/0.5_1.8_ratio_0.5",
                "Epidemic_values/0.5_2.5_ratio_0.25", "Epidemic_values/0.5_2.5_ratio_0.5",
                "Epidemic_values/1.0_1.2_ratio_0.25", "Epidemic_values/1.0_1.2_ratio_0.5",
                "Epidemic_values/1.0_1.8_ratio_0.25", "Epidemic_values/1.0_1.8_ratio_0.5",
                "Epidemic_values/1.0_2.5_ratio_0.25", "Epidemic_values/1.0_2.5_ratio_0.5"
            ]),
            (f"./sens_data/icu/icu_values", [
                "icu_values/0.5_1.2_ratio_0.25", "icu_values/0.5_1.2_ratio_0.5",
                "icu_values/0.5_1.8_ratio_0.25", "icu_values/0.5_1.8_ratio_0.5",
                "icu_values/0.5_2.5_ratio_0.25", "icu_values/0.5_2.5_ratio_0.5",
                "icu_values/1.0_1.2_ratio_0.25", "icu_values/1.0_1.2_ratio_0.5",
                "icu_values/1.0_1.8_ratio_0.25", "icu_values/1.0_1.8_ratio_0.5",
                "icu_values/1.0_2.5_ratio_0.25", "icu_values/1.0_2.5_ratio_0.5",
            ]),
            (f"./sens_data/death/death_values", [
                "death_values/0.5_1.2_ratio_0.25", "death_values/0.5_1.2_ratio_0.5",
                "death_values/0.5_1.8_ratio_0.25", "death_values/0.5_1.8_ratio_0.5",
                "death_values/0.5_2.5_ratio_0.25", "death_values/0.5_2.5_ratio_0.5",
                "death_values/1.0_1.2_ratio_0.25", "death_values/1.0_1.2_ratio_0.5",
                "death_values/1.0_1.8_ratio_0.25", "death_values/1.0_1.8_ratio_0.5",
                "death_values/1.0_2.5_ratio_0.25", "death_values/1.0_2.5_ratio_0.5"
            ]),

            (f"./sens_data/hospital/hospital_values", [
                "hospital_values/0.5_1.2_ratio_0.25", "hospital_values/0.5_1.2_ratio_0.5",
                "hospital_values/0.5_1.8_ratio_0.25", "hospital_values/0.5_1.8_ratio_0.5",
                "hospital_values/0.5_2.5_ratio_0.25", "hospital_values/0.5_2.5_ratio_0.5",
                "hospital_values/1.0_1.2_ratio_0.25", "hospital_values/1.0_1.2_ratio_0.5",
                "hospital_values/1.0_1.8_ratio_0.25", "hospital_values/1.0_1.8_ratio_0.5",
                "hospital_values/1.0_2.5_ratio_0.25", "hospital_values/1.0_2.5_ratio_0.5"
            ])
        ]
        # Define index
        if model == "rost":
            index = ["All", "0-4", "5-9", "10-14", "15-19", "20-24",
                     "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                     "55-59", "60-64", "65-69", "70-74", "75+"]

        elif model == "moghadas":
            index = ["All", "0-19", "20-49", "50-65", "65+"]
        elif model == "chikina":
            index = ["All", "0-4", "5-9", "10-14", "15-19", "20-24",
                     "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                     "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
        elif model == "seir":
            index = ["All", "0-4", "5-9", "10-14", "15-19", "20-24",
                     "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                     "55-59", "60-64", "65-69", "70+"]
        else:
            raise Exception("Invalid model")

        for directory, column_order in directory_column_orders:
            # Concatenate the DataFrames horizontally and reorder columns
            stacked_df = pd.concat([max_values[col] for col in column_order], axis=1).T

            # Add labels to the y-axis
            labels = [fr"$\sigma=0.5$, $\overline{{\mathcal{{R}}}}_0=1.2$, $\mathcal{{R}}=0.25$",
                      fr"$\sigma=0.5$, $\overline{{\mathcal{{R}}}}_0=1.2$, $\mathcal{{R}}=0.5$",
                      fr"$\sigma=0.5$, $\overline{{\mathcal{{R}}}}_0=1.8$, $\mathcal{{R}}=0.25$",
                      fr"$\sigma=0.5$, $\overline{{\mathcal{{R}}}}_0=1.8$, $\mathcal{{R}}=0.5$",
                      fr"$\sigma=0.5$, $\overline{{\mathcal{{R}}}}_0=2.5$, $\mathcal{{R}}=0.25$",
                      fr"$\sigma=0.5$, $\overline{{\mathcal{{R}}}}_0=2.5$, $\mathcal{{R}}=0.5$",
                      fr"$\sigma=1.0$, $\overline{{\mathcal{{R}}}}_0=1.2$, $\mathcal{{R}}=0.25$",
                      fr"$\sigma=1.0$, $\overline{{\mathcal{{R}}}}_0=1.2$, $\mathcal{{R}}=0.5$",
                      fr"$\sigma=1.0$, $\overline{{\mathcal{{R}}}}_0=1.8$, $\mathcal{{R}}=0.25$",
                      fr"$\sigma=1.0$, $\overline{{\mathcal{{R}}}}_0=1.8$, $\mathcal{{R}}=0.5$",
                      fr"$\sigma=1.0$, $\overline{{\mathcal{{R}}}}_0=2.5$, $\mathcal{{R}}=0.25$",
                      fr"$\sigma=1.0$, $\overline{{\mathcal{{R}}}}_0=2.5$, $\mathcal{{R}}=0.5$"]

            plt.figure(figsize=(22, 8))
            heatmap = sns.heatmap(stacked_df, cmap='inferno', annot=True, fmt=".1f")
            # Draw a vertical line after the first column
            plt.axvline(x=1.0, color='white', linestyle="solid", linewidth=10)
            # Remove y-axis label for the inserted column
            heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
            # Rotate x-axis labels
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90,
                                    horizontalalignment='right', fontsize=18,
                                    fontweight='bold')

            plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels,
                       rotation=0, horizontalalignment='right', fontsize=12,
                       fontweight='bold')

            plt.xticks(ticks=np.arange(len(index)) + 0.5, labels=index,
                       rotation=90, horizontalalignment='right', fontsize=12,
                       fontweight='bold')
            if plot_title == 'N':
                heatmap.set_title(os.path.basename(directory), fontsize=35,
                                  fontweight='bold')
            # Save the heatmap under the directory
            save_path = os.path.join(directory, "heatmap.pdf")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

    def plot_epidemic_peak_and_size(self, time, cm_list, legend_list, ratio, model: str,
                                    plot_option: str = "Epidemic_size"):
        """
        Calculates and saves the max values after running an epidemic peak simulation by varying
        age group contacts, then plots the epidemic peak size for different combinations
        of susc, base_ratio, and ratios.
        :param time: time points.
        :param cm_list: (list): List of legends corresponding to each combination.
        :param legend_list:
        :param ratio: The ratio value, [0.25, 0.5]
        :param model: The different models
        :param plot_option: plot cumulative epidemic size or epidemic peak
        :return: plots and age group max epidemic peak size as csv files.
        """
        # Define the output directory for plot
        plot_dir, output_dir = self.create_directories(target="Epidemic")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Get the Blues colormap
        blues_cmap = get_cmap('Blues')
        color = blues_cmap(0.3)
        color2 = blues_cmap(0.5)

        if model == "rost":
            compartments = ["l1", "l2", "ip", "ia1", "ia2", "ia3",
                            "is1", "is2", "is3", "ih", "ic", "icr"]
        elif model == "moghadas":
            compartments = ["e", "i_n",
                            "q_n", "i_h", "q_h", "a_n",
                            "a_q", "h", "c"]
        elif model == "chikina":
            compartments = ["i", "cp", "c"]
        elif model == "seir":
            compartments = ["e", "i"]
        else:
            raise Exception("Invalid model")

        # Initialize an empty list to store results
        results_list = []
        df = pd.DataFrame()

        susc = self.sim_obj.sim_state["susc"]
        base_r0 = self.sim_obj.sim_state["base_r0"]
        max_peak_size_per_compartment = []
        # Initialize combined_curve outside the loop
        combined_curve = np.zeros_like(time)
        init_values = self.sim_obj.model.get_initial_values()
        for cm, legend in zip(cm_list, legend_list):
            # Get solution for the current combination
            solution = self.sim_obj.model.get_solution(
                init_values=init_values,
                t=time,
                parameters=self.sim_obj.params,
                cm=cm
            )
            # Iterate over compartments and sum values
            for compartment in compartments:
                max_peak_size = 0
                for t_idx in range(len(solution)):
                    compartment_values = self.sim_obj.model.aggregate_by_age(
                        solution=solution,
                        idx=self.sim_obj.model.c_idx[compartment])
                    peak_size_at_t = compartment_values.sum()
                    max_peak_size = max(max_peak_size, peak_size_at_t)
                max_peak_size_per_compartment.append(max_peak_size)

                # Calculate the combined epidemic curve
                combined_curve += np.sum(
                    [solution[:, self.sim_obj.model.c_idx[comp] * self.sim_obj.n_ag:
                              (self.sim_obj.model.c_idx[comp] + 1) * self.sim_obj.n_ag
                              ].sum(axis=1) for comp in self.sim_obj.model.compartments], axis=0)
            if model in ["rost", "seir"]:
                total_infecteds = self.sim_obj.model.aggregate_by_age(
                    solution=solution,
                    idx=self.sim_obj.model.c_idx["c"])
            elif model == "moghadas":
                total_infecteds = self.sim_obj.model.aggregate_by_age(
                    solution=solution,
                    idx=self.sim_obj.model.c_idx["i"])
            elif model == "chikina":
                total_infecteds = self.sim_obj.model.aggregate_by_age(
                    solution=solution,
                    idx=self.sim_obj.model.c_idx["inf"])
            else:
                raise Exception("Invalid model")

            # Save the results in a dictionary with metadata
            result_entry = {
                'susc': susc,
                'base_r0': base_r0,
                'ratio': ratio,
                'n_infected_max': total_infecteds,
                'n_infecteds_peak': combined_curve
            }
            # Append the result_entry to the results_list
            results_list.append(result_entry)
            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(results_list)

            # Plot the epidemic peak for the current combination
            if plot_option == "Epidemic_size":
                # Plot the cumulative epidemic size for the current combination
                ax.plot(time, result_entry['n_infected_max'],
                        label=f'susc={susc}, 'f'r0={base_r0}, ratio={1 - ratio}', color=color2)
                ax.fill_between(time, 0, result_entry['n_infected_max'],
                                color=color2, alpha=0.8)
            else:
                ax.plot(time, result_entry['n_infecteds_peak'],
                        label=f'susc={susc}, 'f'r0={base_r0}, ratio={1 - ratio}', color=color2)
                ax.fill_between(time, 0, result_entry['n_infecteds_peak'],
                                color=color2, alpha=0.8)
        # Find the peak value and its corresponding day
        peak_day = np.argmax(combined_curve)
        peak_value = combined_curve[peak_day]
        print(peak_value)
        self.create_rectangular_box_on_the_right(ax=ax, color=color, susc=susc,
                                                 base_r0=base_r0, ratio=ratio,
                                                 model=model,
                                                 plot_option="Epidemic")
        # Set spine properties
        self.set_spine_properties(ax=ax)

        if plot_option == "Epidemic_size":
            output_path = os.path.join(plot_dir,
                                       f"epidemic_size_plot_{susc}_{base_r0}_{ratio}.pdf")
        else:
            output_path = os.path.join(plot_dir,
                                       f"epidemic_peak_plot_{susc}_{base_r0}_{ratio}.pdf")

        plt.savefig(output_path, format="pdf",
                    bbox_inches='tight', pad_inches=0.5)
        plt.savefig(output_path, format="pdf")
        plt.close()

        # Find the maximum value in each row
        max_infected_values = np.array([max(row) for row in df['n_infected_max']])
        # Add a new column 'max_n_icu' to df containing the maximum values
        df['max_n_infected'] = max_infected_values
        # Add an 'age_group' column to your DataFrame based on the index
        df['age_group'] = df.index % (self.sim_obj.contact_matrix.shape[0] + 1)
        # Pivot the DataFrame
        pivot_df = df.pivot_table(index=['susc', 'base_r0', 'ratio'], columns='age_group',
                                  values='max_n_infected',
                                  aggfunc='first')
        # Reset the index and save the results
        self.prepare_and_save_results(pivot_df=pivot_df, susc=susc,
                                      base_r0=base_r0, ratio=ratio,
                                      output_dir=output_dir)

    def plot_solution_final_death_size(self, time, cm_list, legend_list, ratio, model):
        """
        Calculates and saves the max values after simulating the final deaths
        by varying age group contacts, then plots the final death size for different combinations
        of susc, base_ratio, and ratios.
        :param time: time points.
        :param cm_list: (list): List of legends corresponding to each combination.
        :param legend_list:
        :param ratio: The ratio value, [0.25, 0.5]
        :param model: The different models
        :return: plots and age group max final death size as csv files.
        """
        # Define the output directory for plot
        plot_dir, output_dir = self.create_directories(target="death")
        results_list = []

        susc = self.sim_obj.sim_state["susc"]
        base_r0 = self.sim_obj.sim_state["base_r0"]
        init_values = self.sim_obj.model.get_initial_values()
        for cm, legend in zip(cm_list, legend_list):
            solution = self.sim_obj.model.get_solution(
                init_values=init_values,
                t=time,
                parameters=self.sim_obj.params,
                cm=cm)
            deaths = self.sim_obj.model.aggregate_by_age(
                solution=solution,
                idx=self.sim_obj.model.c_idx["d"])
            # Save the results in a dictionary with metadata
            result_entry = {
                'susc': susc,
                'base_r0': base_r0,
                'ratio': ratio,
                'n_deaths_max': deaths
            }
            # Append the result_entry to the results_list
            results_list.append(result_entry)
            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(results_list)
            # Plot the epidemic size for the current combination
            fig, ax = plt.subplots(figsize=(10, 6))
            blues_cmap = get_cmap('Blues')
            color = blues_cmap(0.3)
            color2 = blues_cmap(0.5)
            ax.plot(time, result_entry['n_deaths_max'],
                    label=f'susc={susc}, 'f'r0={base_r0}, ratio={1 - ratio}', color=color2)
            ax.fill_between(time, result_entry['n_deaths_max'],
                            color=color2, alpha=0.8)
            self.create_rectangular_box_on_the_right(ax=ax, color=color, susc=susc,
                                                     base_r0=base_r0, ratio=ratio,
                                                     model=model,
                                                     plot_option="death")
            # Set spine properties
            self.set_spine_properties(ax=ax)
            output_path = os.path.join(plot_dir,
                                       f"death_size_plot_{susc}_{base_r0}_{ratio}.pdf")
            plt.savefig(output_path, format="pdf",
                        bbox_inches='tight', pad_inches=0.5)
            plt.savefig(output_path, format="pdf")
            plt.close()
            # Find the maximum value in each row
            max_death_values = np.array([max(row) for row in df['n_deaths_max']])
            # Add a new column 'max_n_icu' to df containing the maximum values
            df['max_n_deaths'] = max_death_values
            # Add an 'age_group' column to your DataFrame based on the index
            df['age_group'] = df.index % (self.sim_obj.contact_matrix.shape[0] + 1)
            # Pivot the DataFrame
            pivot_df = df.pivot_table(index=['susc', 'base_r0', 'ratio'], columns='age_group',
                                      values='max_n_deaths',
                                      aggfunc='first')
            # Reset the index and save the results
            self.prepare_and_save_results(pivot_df=pivot_df, susc=susc,
                                          base_r0=base_r0, ratio=ratio,
                                          output_dir=output_dir)

    def plot_solution_hospitalized_size(self, time, cm_list, legend_list, ratio, model):
        # Define the output directory for plot
        plot_dir, output_dir = self.create_directories(target="hospital")
        results_list = []
        susc = self.sim_obj.sim_state["susc"]
        base_r0 = self.sim_obj.sim_state["base_r0"]
        init_values = self.sim_obj.model.get_initial_values()
        for cm, legend in zip(cm_list, legend_list):
            solution = self.sim_obj.model.get_solution(
                init_values=init_values,
                t=time,
                parameters=self.sim_obj.params,
                cm=cm)
            # hospital collector
            cum_hospitals = self.sim_obj.model.aggregate_by_age(
                solution=solution, idx=self.sim_obj.model.c_idx["hosp"])

            # Save the results in a dictionary with metadata
            result_entry = {
                'susc': susc,
                'base_r0': base_r0,
                'ratio': ratio,
                'n_hospitals_max': cum_hospitals
            }
            # Append the result_entry to the results_list
            results_list.append(result_entry)

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(results_list)
        fig, ax = plt.subplots(figsize=(8, 6))
        blues_cmap = get_cmap('Blues')
        color = blues_cmap(0.3)
        color2 = blues_cmap(0.5)
        ax.plot(time, df['n_hospitals_max'].iloc[0], color=color2)
        ax.fill_between(time, df['n_hospitals_max'].iloc[0],
                        color=color2, alpha=0.8)
        self.create_rectangular_box_on_the_right(ax=ax, color=color, susc=susc,
                                                 base_r0=base_r0, ratio=ratio,
                                                 model=model,
                                                 plot_option="Hospitalized")
        # Set spine properties
        self.set_spine_properties(ax=ax)
        output_path = os.path.join(plot_dir,
                                   f"hosp_size_plot_{susc}_{base_r0}_{1 - ratio}.pdf")
        plt.savefig(output_path, format="pdf", bbox_inches='tight', pad_inches=0.5)
        plt.close()

        # Find the maximum value in each row
        max_hosp_values = np.array([max(row) for row in df['n_hospitals_max']])
        # Add a new column 'max_n_icu' to df containing the maximum values
        df['max_n_hosp'] = max_hosp_values
        # Add an 'age_group' column to your DataFrame based on the index
        df['age_group'] = df.index % (self.sim_obj.contact_matrix.shape[0] + 1)
        # Pivot the DataFrame
        pivot_df = df.pivot_table(index=['susc', 'base_r0', 'ratio'], columns='age_group',
                                  values='max_n_hosp',
                                  aggfunc='first')
        # Reset the index and save the results
        self.prepare_and_save_results(pivot_df=pivot_df, susc=susc,
                                      base_r0=base_r0, ratio=ratio,
                                      output_dir=output_dir)

    def plot_icu_size(self, time, cm_list, legend_list, ratio, model: str):
        """
        Calculates and saves the max values after simulating the icu individuals
        by varying age group contacts, then plots the icu size for different combinations
        of susc, base_ratio, and ratios.
        :param time: time points.
        :param cm_list: (list): List of legends corresponding to each combination.
        :param legend_list:
        :param ratio: The ratio value, [0.25, 0.5]
        :param model: The different models
        :return: plots and age group max icu size as csv files.
        """
        # Define the output directory for plot
        plot_dir, output_dir = self.create_directories(target="icu")
        susc = self.sim_obj.sim_state["susc"]
        base_r0 = self.sim_obj.sim_state["base_r0"]
        results_list = []
        initial_values = self.sim_obj.model.get_initial_values()
        for cm, legend in zip(cm_list, legend_list):
            solution = self.sim_obj.model.get_solution(
                init_values=initial_values,
                t=time,
                parameters=self.sim_obj.params,
                cm=cm)
            # icu collector
            cum_icu = self.sim_obj.model.aggregate_by_age(
                    solution=solution, idx=self.sim_obj.model.c_idx["icu"])
            # Save the results in a dictionary with metadata
            result_entry = {
                    'susc': susc,
                    'base_r0': base_r0,
                    'ratio': ratio,
                    'n_icu_max': cum_icu
            }
            # Append the result_entry to the results_list
            results_list.append(result_entry)
            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(results_list)
            # Plot the epidemic size for the current combination
            fig, ax = plt.subplots(figsize=(8, 6))
            blues_cmap = get_cmap('Blues')
            color = blues_cmap(0.3)
            color2 = blues_cmap(0.5)
            ax.plot(time, result_entry['n_icu_max'], color=color2)
            ax.fill_between(time, result_entry['n_icu_max'],
                            color=color2, alpha=0.8)
            self.create_rectangular_box_on_the_right(ax=ax, color=color, susc=susc,
                                                     base_r0=base_r0, ratio=ratio,
                                                     model=model,
                                                     plot_option="ICU")
            self.set_spine_properties(ax=ax)
            output_path = os.path.join(plot_dir,
                                       f"ICU_size_plot_{susc}_{base_r0}_{1 - ratio}.pdf")
            plt.savefig(output_path, format="pdf",
                        bbox_inches='tight', pad_inches=0.5)
            plt.savefig(output_path, format="pdf")
            plt.close()
            plt.show()
            # Find the maximum value in each row
            max_icu_values = np.array([max(row) for row in df['n_icu_max']])
            # Add a new column 'max_n_icu' to df containing the maximum values
            df['max_n_icu'] = max_icu_values
            # Add an 'age_group' column to your DataFrame based on the index
            df['age_group'] = df.index % (self.sim_obj.contact_matrix.shape[0] + 1)
            # Pivot the DataFrame
            pivot_df = df.pivot_table(index=['susc', 'base_r0', 'ratio'], columns='age_group',
                                      values='max_n_icu',
                                      aggfunc='first')
            # Reset the index and save the results
            self.prepare_and_save_results(pivot_df=pivot_df, susc=susc,
                                          base_r0=base_r0, ratio=ratio,
                                          output_dir=output_dir)

    @staticmethod
    def prepare_and_save_results(pivot_df, susc, base_r0, ratio, output_dir):
        # Reset the index
        pivot_df = pivot_df.reset_index()
        # Save the entire DataFrame to a CSV file
        fname = "_".join([str(susc), str(base_r0), f"ratio_{1 - ratio}"])
        filename = os.path.join(output_dir, fname + ".csv")
        np.savetxt(fname=filename, X=np.sum(pivot_df)[3:], delimiter=";")

    @staticmethod
    def create_directories(target):
        # Define the base directory for sensitivity analysis data
        base_dir = "./sens_data"
        # Define the output directory for plot
        plot_dir = os.path.join(base_dir, target, "plot")
        os.makedirs(plot_dir, exist_ok=True)
        # Define the output directory for maximum values of the outcome
        output_dir = os.path.join(base_dir, target, target + "_values")
        os.makedirs(output_dir, exist_ok=True)
        return plot_dir, output_dir

    @staticmethod
    def create_rectangular_box_on_the_right(ax, color, susc, base_r0, ratio,
                                            model,
                                            plot_option):
        # Get y-axis limits
        y_min, y_max = ax.get_ylim()
        # Calculate patch height based on the axis limits
        patch_height = y_max - y_min

        # Define x_start, adj, and text_adj based on the model
        if model == "rost":  # t = 1200
            x_start = 1200
            adj = 200
            text_adj = 35
        elif model == "moghadas":  # t = 400
            x_start = 400
            adj = 50
            text_adj = 10
        elif model == "chikina":  # t = 2500
            x_start = 2500
            adj = 150
            text_adj = 60
        elif model == "seir":  # t = 200
            x_start = 400
            adj = 50
            text_adj = 10
        else:
            raise Exception("Invalid model")
        # Create a rectangular box for the y-axis label on the right side
        rect = patches.Rectangle((x_start, y_min), width=adj, height=patch_height,
                                 color=color,
                                 linewidth=1, alpha=0.8, edgecolor=color,  # Set the edge color for the border
                                 linestyle='solid')
        # Add vertical line at the start of the rectangle
        line = lines.Line2D([x_start, x_start], [y_min, y_max], color='black',
                            linewidth=1.2)
        if plot_option == "ICU":
            ax.text(x_start + text_adj, y_min + 0.5 * patch_height,
                    'Cumulative ICU size population',
                    rotation=90, verticalalignment='center',
                    horizontalalignment='center', fontsize=12, color='navy',
                    weight='bold'
                    )
        elif plot_option == "Hospitalized":
            ax.text(x_start + text_adj, y_min + 0.5 * patch_height,
                    'Cumulative hospitalized size population',
                    rotation=90, verticalalignment='center',
                    horizontalalignment='center', fontsize=12, color='navy',
                    weight='bold'
                    )
        elif plot_option == "death":
            ax.text(x_start + text_adj, y_min + 0.5 * patch_height,
                    'Cumulative Death size population',
                    rotation=90, verticalalignment='center',
                    horizontalalignment='center', fontsize=12, color='navy',
                    weight='bold'
                    )
        elif plot_option == "Epidemic":
            ax.text(x_start + text_adj, y_min + 0.5 * patch_height,
                    'Cumulative Epidemic size population',
                    rotation=90, verticalalignment='center',
                    horizontalalignment='center', fontsize=12, color='navy',
                    weight='bold'
                    )
        # Construct legend text
        legend_text = f'${{\\sigma={susc}}}$, ${{\\overline{{\\mathcal{{R}}}}_0={base_r0}}}$, ' \
                      f'${{\\mathcal{{R}}={1 - ratio}}}$'
        # Plot the legend
        ax.legend([TriangleHandler()], [legend_text], fontsize=10, labelcolor="navy",
                  loc='upper left', bbox_to_anchor=(0.0, 1.0))

        ax.add_patch(rect)
        ax.add_line(line)

    @staticmethod
    def set_spine_properties(ax):
        # Set spine properties
        ax.spines['top'].set_linewidth(0.8)
        ax.spines['top'].set_color("grey")
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['bottom'].set_color("grey")
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['left'].set_color("grey")
        ax.spines['right'].set_linewidth(1.2)
        ax.spines['right'].set_color("black")
        ax.set_xlabel("day", fontsize=18)


class TriangleHandler(Line2D):
    def __init__(self, **kwargs):
        # Default line properties
        line_props = {'linewidth': 0, 'linestyle': '-'}

        # Extract label and color from kwargs
        label = kwargs.pop('label', None)
        color = kwargs.pop('color', 'navy')

        # Call the parent constructor
        super().__init__([], [], label=label, color=color, marker='^',
                         markersize=10, **line_props)
