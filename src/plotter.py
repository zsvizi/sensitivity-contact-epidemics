import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import LogLocator, LogFormatterSciNotation as LogFormatter
import matplotlib.colors as colors

from matplotlib.tri import Triangulation
import numpy as np
import pandas as pd

from src.dataloader import DataLoader
from src.simulation_npi import SimulationNPI
from src.prcc import get_rectangular_matrix_from_upper_triu
plt.style.use('seaborn-whitegrid')


class Plotter:
    def __init__(self, sim_obj: SimulationNPI) -> None:
        self.data = DataLoader()
        self.sim_obj = sim_obj

    def plot_contact_matrices_hungary(self, filename):
        os.makedirs("sens_data/contact_matrices", exist_ok=True)
        cols = {"home": "PRGn", "work": "Paired_r", "school": "RdYlGn_r", "other": "PiYG_r",
                  "total": "gist_earth_r"}
        cmaps = {"home": "gist_earth_r", "work": "gist_earth_r", "school": "gist_earth_r",
                 "other": "gist_earth_r", "total": "gist_earth_r"}
        contact_full = np.array([self.data.contact_data[i] for i in list(cols.keys())[:-1]]).sum(axis=0)
        for i in cols.keys():
            contacts = self.data.contact_data[i] if i != "total" else contact_full
            param_list = range(0, 16, 1)
            contact_matrix = pd.DataFrame(contacts, columns=param_list, index=param_list)
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif', size=14)
            plot = plt.imshow(contact_matrix, cmap=cmaps[i], origin='lower',
                              alpha=.9, interpolation="nearest", vmin=0, vmax=5)
            number_of_age_groups = 16
            age_groups = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
                          "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75+"]
            plt.gca().set_xticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
            plt.gca().set_yticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
            plt.gca().grid(which='minor', color='gray', linestyle='-', linewidth=1)
            plt.xticks(ticks=param_list, rotation=0, fontsize=15)
            plt.yticks(ticks=param_list, fontsize=15)
            if i == "total":
                cbar = plt.colorbar(plot, pad=0.02, fraction=0.04)
                tick_font_size = 40
                cbar.ax.tick_params(labelsize=tick_font_size)
                plt.gca().set_xticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
                plt.gca().set_yticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
                plt.gca().grid(which='minor', color='gray', linestyle='-', linewidth=1)
                plt.xticks(ticks=param_list, rotation=0, fontsize=15)
                plt.yticks(ticks=param_list, fontsize=15)
            plt.title(i + " contact", y=1.03, fontsize=25)
            plt.savefig('./sens_data/contact_matrices/' + filename + "_" + i + '.pdf',
                         format="pdf", bbox_inches='tight')
            plt.close()

    def plot_2d_contact_matrices_total(self, filename):
        os.makedirs("sens_data/contact_total", exist_ok=True)
        colors = {"home": "#ff96da", "work": "#96daff", "school": "#96ffbb",
                  "other": "#ffbb96", "total": "Set2"}
        cmaps = {"home": "CMRmap_r", "work": "summer", "school": "summer_r", "other": "gist_stern_r",
                 "total": "Dark2"}
        contact_total = np.array([self.data.contact_data[t] for t in list(colors.keys())[:-1]]).sum(axis=0)
        for t in colors.keys():
            contact_small = self.data.contact_data[t] if t != "total" else contact_total
            param_list = range(0, 16, 1)
            corr = pd.DataFrame(contact_small * self.data.age_data, columns=param_list, index=param_list)
            plt.figure(figsize=(12, 12))
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif', size=14)
            plt.title('Total ' + t + ' contact', y=1.03, fontsize=25)
            plt.imshow(corr, cmap=cmaps[t], origin='lower')
            plt.colorbar(pad=0.02, fraction=0.04)
            # plt.grid(b=None)
            number_of_age_groups = 16
            age_groups = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
                          "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75+"]

            plt.gca().set_xticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
            plt.gca().set_yticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
            plt.gca().grid(which='minor', color='gray', linestyle='-', linewidth=1)
            plt.xticks(ticks=param_list, labels=age_groups, rotation=90)
            plt.yticks(ticks=param_list, labels=age_groups)
            plt.savefig('./sens_data/contact_total/' + filename + "_" + t + '.pdf',
                        format="pdf", bbox_inches='tight')
            plt.close()

    def plot_prcc_p_values_as_heatmap(self, prcc, p_values, filename_without_ext,
                                      filename_to_save, plot_title):
        os.makedirs("sens_data/heatmap", exist_ok=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.margins(0, tight=False)
        PRC_mtx = get_rectangular_matrix_from_upper_triu(prcc[:self.sim_obj.upper_tri_size],
                                                         self.sim_obj.n_ag)
        P_values_mtx = get_rectangular_matrix_from_upper_triu(p_values[:self.sim_obj.upper_tri_size],
                                                              self.sim_obj.n_ag)
        # vertices of the little squares
        xv, yv = np.meshgrid(np.arange(-0.5, self.sim_obj.n_ag), np.arange(-0.5, self.sim_obj.n_ag))
        # centers of the little square
        xc, yc = np.meshgrid(np.arange(0, self.sim_obj.n_ag), np.arange(0, self.sim_obj.n_ag))
        x = np.concatenate([xv.ravel(), xc.ravel()])
        y = np.concatenate([yv.ravel(), yc.ravel()])
        start = (self.sim_obj.n_ag + 1) * (self.sim_obj.n_ag + 1)  # indices of the centers
        trianglesPRCC = [(i + j * (self.sim_obj.n_ag + 1), i + 1 + j * (self.sim_obj.n_ag + 1),
                          i + (j + 1) * (self.sim_obj.n_ag + 1))
                         for j in range(self.sim_obj.n_ag) for i in range(self.sim_obj.n_ag)]
        trianglesP = [(i + 1 + j * (self.sim_obj.n_ag + 1), i + 1 + (j + 1) * (self.sim_obj.n_ag + 1),
                       i + (j + 1) * (self.sim_obj.n_ag + 1))
                      for j in range(self.sim_obj.n_ag) for i in range(self.sim_obj.n_ag)]
        triang = [Triangulation(x, y, triangles, mask=None)
                  for triangles in [trianglesPRCC, trianglesP]]

        values_all = [PRC_mtx, P_values_mtx]
        values = np.triu(values_all, k=0)
        mask = np.where(values[0] == 0, np.nan, values_all)
        p_value_cmap = ListedColormap(['Orange', 'Coral', 'red', 'darkred'])
        cmaps = ["Greens", p_value_cmap]

        log_norm = colors.LogNorm(vmin=1e-4, vmax=1e0)    # used for p_values
        norm = plt.Normalize(vmin=0, vmax=1)  # used for PRCC_values

        fig, ax = plt.subplots()
        images = [ax.tripcolor(t, np.ravel(val), cmap=cmap, ec="white")
                  for t, val, cmap in zip(triang, mask, cmaps)]

        cbar = fig.colorbar(images[0], ax=ax, shrink=0.7, aspect=20 * 0.7)   # for the prcc values
        cbar_pval = fig.colorbar(images[1], ax=ax, shrink=0.7, aspect=20 * 0.7)

        images[1].set_norm(norm=log_norm)
        images[0].set_norm(norm=norm)

        locator = LogLocator()
        formatter = LogFormatter()
        cbar_pval.locator = locator
        cbar_pval.formatter = formatter
        cbar_pval.update_normal(images[1])

        ax.set_xticks(range(self.sim_obj.n_ag))
        ax.set_yticks(range(self.sim_obj.n_ag))
        plt.gca().grid(which='minor', color='gray', linestyle='-', linewidth=1)
        ax.margins(x=0, y=0)
        ax.set_aspect('equal', 'box')  # square cells
        plt.tight_layout()
        title_list = filename_without_ext.split("_")
        plt.title(plot_title, y=1.03, fontsize=25)
        plt.title(plot_title, y=1.03, fontsize=25)
        plt.savefig('./sens_data/heatmap/' + filename_to_save + '.pdf', format="pdf", bbox_inches='tight')
        plt.close()

    def generate_prcc_p_values_heatmaps(self, prcc_vector, p_values, filename_without_ext, target:
                                        str = "Final death size"):
        title_list = filename_without_ext.split("_")
        plot_title = 'Target:' + target + ', Susceptibility=' + title_list[0] + ', R0=' + title_list[1]
        self.plot_prcc_p_values_as_heatmap(prcc_vector, p_values, filename_without_ext,
                                           "PRCC_P_VALUES" + filename_without_ext + "_R0", plot_title)

    def aggregated_prcc_pvalues_plots(self, param_list, prcc_vector, p_values, filename_without_ext,
                             filename_to_save, plot_title):
        os.makedirs("sens_data/PRCC_PVAL_PLOT", exist_ok=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.margins(0, tight=False)
        xp = range(param_list)
        plt.figure(figsize=(15, 12))
        plt.tick_params(direction="in")
        fig, ax = plt.subplots()
        plt.bar(xp, list(abs(prcc_vector)), align='center', width=0.8, alpha=0.5, color="g", label="PRCC")
        for pos, y, err in zip(xp, list(abs(prcc_vector)), list(abs(p_values))):
            plt.errorbar(pos, y, err, lw=4, capthick=4, fmt="or",
                         markersize=5, capsize=4, ecolor="r", elinewidth=4,
                        color='b')
        plt.xticks(ticks=xp, rotation=90)
        plt.yticks(ticks=np.arange(-1, 1.2, 0.2))
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0, 1.2])
        plt.ylabel('Aggregated PRCC', labelpad=10, fontsize=20)
        plt.xlabel('Pairs of age groups', labelpad=10, fontsize=20)
        plt.title("PRCC values and their corresponding P values")
        title_list = filename_without_ext.split("_")
        plt.title(plot_title, y=1.03, fontsize=20)
        plt.savefig('./sens_data/PRCC_PVAL_PLOT/' + filename_to_save + '.pdf',
        format="pdf", bbox_inches='tight')
        plt.close()

    def plot_aggregation_prcc_pvalues(self, prcc_vector, p_values, filename_without_ext,
                                      target: str = "R0"):
        title_list = filename_without_ext.split("_")
        plot_title = 'Target:' + target + ', Susceptibility=' + title_list[0] + ', R0=' + title_list[1]
        self.aggregated_prcc_pvalues_plots(16, prcc_vector, p_values, filename_without_ext,
                                   "PRCC_P_VALUES_" + filename_without_ext + "_R0", plot_title)
