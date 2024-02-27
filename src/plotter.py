import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import LogLocator, LogFormatterSciNotation as LogFormatter
import matplotlib.colors as colors
import matplotlib.collections as collections
from cycler import cycler
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.tri import Triangulation
import numpy as np
import pandas as pd
import seaborn as sns

from src.dataloader import DataLoader
from src.simulation_base import SimulationBase
from src.prcc import get_rectangular_matrix_from_upper_triu

plt.style.use('seaborn-whitegrid')


class Plotter:
    def __init__(self, data: DataLoader, sim_obj: SimulationBase) -> None:

        self.deaths = None
        self.hospitalized = None

        self.data = data
        self.sim_obj = sim_obj
        self.f1 = np.array([])
        self.f2 = np.array([])
        self.f3 = np.array([])
        self.f4 = np.array([])
        self.f5 = np.array([])
        self.f6 = np.array([])
        self.contact_full = np.array([])

    def plot_contact_matrices_hungary(self, filename, model: str):
        output_dir = "contact_matrices"
        os.makedirs(output_dir, exist_ok=True)
        models = ['seir', 'chikina', 'moghadas', 'rost']

        cmaps = {
            "chikina": {"Home": "viridis", "Work": "viridis", "School": "viridis", "Other": "viridis",
                        "Full": "viridis"},
            "moghadas": {"Home": "viridis", "All": "viridis"},
            "seir": {"physical_contact": "viridis", "all_contact": "viridis"},
            "rost": {"Home": "viridis", "Work": "viridis", "School": "viridis", "Other": "viridis", "Full": "viridis"}
        }

        index_labels = {
            "chikina": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                        "55-59", "60-64", "65-69", "70-74", "75-79", "80+"],
            "moghadas": ["0-19", "20-49", "50-65", "65+"],
            "seir": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                     "55-59", "60-64", "65-69", "70+"],
            "rost": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                     "55-59", "60-64", "65-69", "70-74", "75+"]
        }
        # Iterate over models
        for model in models:
            plt.figure(figsize=(12, 10))

            # Iterate over contact matrices
            for i, key in enumerate(self.data.contact_data.keys()):
                contacts = self.data.contact_data[key]
                param_list = range(len(contacts))

                # Plot heatmap
                plt.subplot(2, 3, i + 1)
                plt.imshow(contacts, cmap=cmaps[model].get(key, 'viridis'), origin='lower', alpha=0.9,
                           interpolation='nearest', vmin=0, vmax=10)

                # Add color bar only for the last contact matrix
                if i == len(self.data.contact_data) - 1:
                    cbar = plt.colorbar(pad=0.02, fraction=0.04)
                    tick_font_size = 12
                    cbar.ax.tick_params(labelsize=tick_font_size)

                number_of_age_groups = len(contacts)
                plt.gca().set_xticks(np.arange(0, number_of_age_groups, 1))
                plt.gca().set_yticks(np.arange(0, number_of_age_groups, 1))
                plt.xticks(ticks=param_list, labels=index_labels[model],
                           rotation=90, fontsize=10, fontweight="bold")
                plt.yticks(ticks=param_list, labels=index_labels[model],
                           fontsize=10, fontweight="bold")
                plt.title(f'{key} contact', y=1.03, fontsize=12, fontweight="bold")

            # Adjust layout and save the figure
            plt.tight_layout()

            # Save the figure in the specified directory
            output_path = os.path.join(output_dir, f'heatmap_{model}.pdf')
            plt.savefig(output_path, format='pdf')

            # Close the figure
            plt.close()


    def plot_contact_matrices_hungary2(self, filename):
        os.makedirs("sens_data/contact_matrices", exist_ok=True)
        cmaps = {"Home": "viridis", "Work": "viridis", "School": "viridis",
                 "Other": "viridis", "Full": "viridis"}
        labels = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
         "55-59", "60-64", "65-69", "70-74", "75+"]
        contact_full = np.array([self.data.contact_data[i]
                                 for i in list(cmaps.keys())[:-1]]).sum(axis=0)
        self.contact_full = contact_full
        for i in cmaps.keys():
            contacts = self.data.contact_data[i] if i != "Full" else contact_full
            param_list = range(0, self.sim_obj.n_ag, 1)
            contact_matrix = pd.DataFrame(contacts, columns=param_list, index=param_list)
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif', size=14)
            plot = plt.imshow(contact_matrix, cmap=cmaps[i], origin='lower',
                              alpha=.9, interpolation="nearest", vmin=0, vmax=10)
            number_of_age_groups = self.sim_obj.n_ag
            plt.gca().set_xticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
            plt.gca().set_yticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
            plt.gca().grid(which='minor', color='gray', linestyle='-', linewidth=1)
            plt.xticks(ticks=param_list, labels=labels,
                       rotation=90, fontsize=12, fontweight="bold")
            plt.yticks(ticks=param_list, labels=labels,
                       fontsize=12, fontweight="bold")
            if i == "Full":
                cbar = plt.colorbar(plot, pad=0.02, fraction=0.04)
                tick_font_size = 40
                cbar.ax.tick_params(labelsize=tick_font_size)
            plt.title(i + " contact", y=1.03, fontsize=25, fontweight="bold")
            plt.savefig('./sens_data/contact_matrices/' + filename + "_" + i + '.pdf',
                        format="pdf", bbox_inches='tight')
            plt.close()

    def get_plot_hungary_heatmap(self):
        os.makedirs("sens_data/contact_matrices", exist_ok=True)
        plt.figure(figsize=(30, 20))
        cols = {"Home": 'jet', "Work": 'jet', "School": 'jet', "Other": 'jet',
                "Full": 'jet'}

        sns.set(font_scale=2.5)
        ax = sns.heatmap(self.contact_full, cmap="Greens", annot=True, square=True,
                         linecolor='white', linewidths=1, cbar=False)

        ax.invert_yaxis()
        # ax.yaxis.set_label_position("right")
        plt.xlabel("Age of the participant", fontsize=70)
        plt.ylabel("Age of the contact", fontsize=70)
        plt.savefig('./sens_data/contact_matrices/' + 'hungary.pdf',
                    format="pdf", bbox_inches='tight')

        # plot mask heatmap
        mask = np.triu(np.ones_like(self.contact_full), k=1).astype(bool)
        fig = plt.figure(figsize=(30, 20))
        ax1 = fig.add_subplot(111)
        cmap = plt.cm.get_cmap('Greens', 10)

        cmap.set_bad('w')  # default value is 'k'
        sns.set(font_scale=2.5)
        sns.heatmap(self.contact_full, mask=mask, annot=True,
                    cmap=cmap, cbar=False, square=True,
                    linecolor='white', linewidths=1)
        ax1.xaxis.set_label_position("top")
        ax1.xaxis.tick_top()
        ax1.invert_yaxis()
        plt.xlabel("Age of the participant", fontsize=70)
        plt.ylabel("Age of the contact", fontsize=70)
        plt.savefig('./sens_data/contact_matrices/' + 'hungary_mask.pdf',
                    format="pdf", bbox_inches='tight')

        plt.close()

    def construct_triangle_grids_prcc_p_value(self, prcc_vector, p_values):
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
                                        p_values, filename_to_save, plot_title):
        p_value_cmap = ListedColormap(['Orange', 'red', 'darkred'])
        # p_valu = ListedColormap(['black', 'black', 'violet', 'violet', 'purple', 'grey',
        #                          'lightgreen',
        #                          'green', 'green', 'darkgreen'])
        cmaps = ["Greens", p_value_cmap]

        log_norm = colors.LogNorm(vmin=1e-3, vmax=1e0)  # used for p_values
        norm = plt.Normalize(vmin=0, vmax=1)  # used for PRCC_values
        # norm = colors.Normalize(vmin=-1, vmax=1e0)  # used for p_values

        fig, ax = plt.subplots()
        triang = self.construct_triangle_grids_prcc_p_value(prcc_vector=prcc_vector,
                                                            p_values=p_values)
        mask = self.get_mask_and_values(prcc_vector=prcc_vector, p_values=p_values)
        images = [ax.tripcolor(t, np.ravel(val), cmap=cmap, ec="white")
                  for t, val, cmap in zip(triang,
                                          mask, cmaps)]

        # fig.colorbar(images[0], ax=ax, shrink=0.7, aspect=20 * 0.7)  # for the prcc values
        cbar = fig.colorbar(images[0], ax=ax, shrink=0.7, aspect=20 * 0.7, pad=0.1)

        cbar_pval = fig.colorbar(images[1], ax=ax, shrink=0.7, aspect=20 * 0.7, pad=0.1)

        images[1].set_norm(norm=log_norm)
        # images[0].set_norm(norm=norm)
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
        plt.savefig('./sens_data/heatmap/' + filename_to_save + '.pdf', format="pdf",
                    bbox_inches='tight')
        plt.close()

    def generate_prcc_p_values_heatmaps(self, prcc_vector,
                                        p_values, filename_without_ext):
        os.makedirs("sens_data/heatmap", exist_ok=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.margins(0, tight=False)
        title_list = filename_without_ext.split("_")
        plot_title = '$\overline{\mathcal{R}}_0=$' + title_list[1]
        self.plot_prcc_p_values_as_heatmap(prcc_vector=prcc_vector, p_values=p_values,
                                           filename_to_save="PRCC_P_VALUES" +
                                                            filename_without_ext + "_R0",
                                           plot_title=plot_title)

    @staticmethod
    def aggregated_prcc_pvalues_plots(param_list, prcc_vector, std_values,
                                      filename_to_save, plot_title):
        os.makedirs("sens_data/PRCC_PVAL_PLOT", exist_ok=True)
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
        plt.savefig('./sens_data/PRCC_PVAL_PLOT/' + filename_to_save + '.pdf',
                    format="pdf", bbox_inches='tight')
        plt.close()

    def plot_aggregation_prcc_pvalues(self, prcc_vector, std_values,
                                      filename_without_ext):
        title_list = filename_without_ext.split("_")
        plot_title = '$\overline{\mathcal{R}}_0=$' + title_list[1]
        self.aggregated_prcc_pvalues_plots(param_list=self.sim_obj.n_ag,
                                           prcc_vector=prcc_vector, std_values=std_values,
                                           filename_to_save=filename_without_ext,
                                           plot_title=plot_title)

    def tornado_plot(self, prcc_vector, std_values,
                                      filename_to_save, plot_title,
                     if_plot_error_bars = "Yes"):
        """
        Creates a tornado plot for sensitivity analysis.

        Parameters:
            labels (list): List of labels for each range.
            base_values (list): List of sensitivity values corresponding to each range.
            p_values (list): List of p-values corresponding to each range.
        """
        os.makedirs("sens_data/PRCC_tornado", exist_ok=True)
        labels = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                  "55-59", "60-64", "65-69", "70-74", "75+"]
        num_params = len(labels)
        y_pos = np.arange(num_params)
        bar_width = 0.4  # Width of each bar

        fig, ax = plt.subplots(figsize=(10, 8))
        if if_plot_error_bars:
            # Plotting sensitivity values
            ax.barh(y_pos, prcc_vector, height=bar_width, align='center', color='green',
                    label='aggregated PRCC')

            # Adding error bars for p-values
            ax.errorbar(prcc_vector, y_pos, xerr=std_values, fmt='none', color='red',
                        label='std-dev', capsize=5)
        else:
            # Plotting base values and p-values side by side
            ax.barh(y_pos - bar_width / 2, prcc_vector, height=bar_width, align='center',
                    color='green',
                    label='aggregated PRCC')
            ax.barh(y_pos + bar_width / 2, p_values, height=bar_width, align='center',
                    color='red', label='std_dev')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # Invert y-axis to have the most important range at the top

        ax.set_xlabel('Aggregated PRCC Values and Standard Deviations')
        plt.title(plot_title, y=1.03, fontsize=20)
        # ax.set_title('Tornado Plot for Sensitivity Analysis')
        ax.legend()
        plt.savefig('./sens_data/PRCC_tornado/' + filename_to_save + '.pdf',
                    format="pdf", bbox_inches='tight')
        plt.close()

    def generate_tornado_plot(self, prcc_vector, std_values, filename_without_ext,
                              if_plot_error_bars="Yes"):
        title_list = filename_without_ext.split("_")
        plot_title = '$\overline{\mathcal{R}}_0=$' + title_list[1]
        self.tornado_plot(prcc_vector=prcc_vector, std_values=std_values,
                          filename_to_save=filename_without_ext,
                          plot_title=plot_title)

    def get_deaths_hospitalized_population(self):
        saved_values = ["age_hospitalized", "age_deaths"]
        for values in saved_values:
            for root, dirs, files in os.walk("./sens_data/" + values):
                for filename in files:
                    filename_without_ext = os.path.splitext(filename)[0]
                    # load the mortality values
                    saved_files = np.loadtxt("./sens_data/" + values + "/" + filename,
                                         delimiter=';')
                    susc = filename.split("_")[3]
                    base_r0 = filename.split("_")[4]
                    base_r0 = base_r0[0:3]

                    if susc == str(0.5) and base_r0 == str(1.2) in filename_without_ext:
                        self.f1 = saved_files  # only the file with susc = 0.5 and base_r0 = 1.2
                    elif susc == str(1.0) and base_r0 == str(2.5) in filename_without_ext:
                        self.f6 = saved_files  # only the file with susc = 1.0 and base_r0 = 2.5
                    elif susc == str(0.5) and base_r0 == str(2.5) in filename_without_ext:
                        self.f3 = saved_files  # only the file with susc = 0.5 and base_r0 = 1.8
                    elif susc == str(1.0) and base_r0 == str(1.2) in filename_without_ext:
                        self.f4 = saved_files  # only the file with susc = 1.0 and base_r0 = 1.2
                    elif susc == str(1.0) and base_r0 == str(1.8) in filename_without_ext:
                        self.f5 = saved_files  # only the file with susc = 1.0 and base_r0 = 1.8
                    elif susc == str(0.5) and base_r0 == str(1.8) in filename_without_ext:
                        self.f2 = saved_files  # only the file with susc = 0.5 and base_r0 = 1.8
            results = pd.DataFrame({
                "$\sigma=0.5$, $\overline{\mathcal{R}}_0=1.2$": self.f1,
                "$\sigma=0.5$, $\overline{\mathcal{R}}_0=1.8$": self.f2,
                "$\sigma=0.5$, $\overline{\mathcal{R}}_0=2.5$": self.f3,
                "$\sigma=1.0$, $\overline{\mathcal{R}}_0=1.2$": self.f4,
                "$\sigma=1.0$, $\overline{\mathcal{R}}_0=1.8$": self.f5,
                "$\sigma=1.0$, $\overline{\mathcal{R}}_0=2.5$": self.f6
            })
            index = ["age 0", "age 1", "age 2", "age 3", "age 4", "age 5", "age 6",
                     "age 7", "age 8", "age 9", "age 10", "age 11", "age 12",
                     "age 13", "age 14", "age 15"]
            results.index = index
            results = results.T
            if values == "age_deaths":
                self.deaths = results
            else:
                self.hospitalized = results

    def plot_horizontal_bars(self):
        os.makedirs("sens_data/hosp_death", exist_ok=True)
        plt.tick_params(direction="in")
        plt.tick_params(direction="in")
        color = ['yellow', 'gold', '#ece75f',  # children
                 'plum', 'violet', 'purple',  # young adults
                 'tomato', '#ff0000', 'crimson', 'darkred',  # middle adults
                 '#ADD8E6', '#89CFF0', '#6495ED',  # older adults
                 '#98FB98', '#50C878', 'green']  # elderly adults

        self.deaths.plot(kind='barh', stacked=True,
                    color=color)

        plt.legend(bbox_to_anchor=(1, 1), loc="upper left", title="age groups")
        plt.xlabel("Age groups distribution")
        axes = plt.gca()
        plt.xticks(ticks=np.arange(0, 1, 0.1))
        axes.set_xlim([0, 1])
        plt.savefig('./sens_data/hosp_death/' + 'final_death.pdf',
                    format="pdf", bbox_inches='tight')
        plt.close()

    def plot_deaths_hospitalized_heatmaps(self):
        os.makedirs("sens_data/hosp_death", exist_ok=True)
        plt.figure(figsize=(30, 20))
        fig, ax = plt.subplots()
        sns.set(font_scale=1.5)
        ax = sns.heatmap(self.hospitalized, cmap="Greens", annot=False, square=True, ax=ax,
                         linecolor='white', linewidths=1, cbar=False)
        ax.grid(True)
        ax.invert_yaxis()
        plt.savefig('./sens_data/hosp_death/' + 'final_hosp.pdf',
                    format="pdf", bbox_inches='tight')
        plt.close()

    @staticmethod
    def triangle_construction_pos(pos=(0, 0), rot=0):
        r = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1]]) * .5
        rm = [[np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot))],
              [np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot))]]
        r = np.dot(rm, r.T).T
        r[:, 0] += pos[0]
        r[:, 1] += pos[1]
        return r

    def triangle_matrix_to_construct(self, a, ax, rot=0, **kwargs):
        seg = []
        for i in range(6):
            for j in range(16):
                seg.append(self.triangle_construction_pos((j, i), rot=rot))
        col = collections.PolyCollection(seg, **kwargs)
        # col.set_array(a.flatten())
        ax.add_collection(col)
        return col

    def plot_death_hospital_heatmap(self):
        os.makedirs("sens_data/hosp_death", exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        A, B = np.meshgrid(range(16), range(6))
        B *= 4
        A *= 4

        # fig, ax = plt.subplots()
        norm = plt.Normalize(vmin=0, vmax=1.0)

        im1 = self.triangle_matrix_to_construct(self.deaths.values,
                                                ax, rot=0, norm=norm, cmap="Greens")
        im2 = self.triangle_matrix_to_construct(self.hospitalized.values, ax,
                                                rot=180, norm=norm, cmap="Greens")
        ax.set_xlim(-.5, A.shape[1] - .5)
        ax.set_ylim(-.5, A.shape[0] - .5)
        im1.set_norm(norm=norm)
        im2.set_norm(norm=norm)
        ax.set_xticks(range(17), labels=self.deaths.columns, rotation=90,
                      fontsize=20)
        ax.set_yticks(range(6), labels=self.deaths.index, fontsize=20)
        fig.colorbar(im1, ax=ax, shrink=0.7, aspect=20 * 0.7, pad=0.01)
        plt.savefig('./sens_data/hosp_death/' + 'hosp_death.pdf', format="pdf",
                    bbox_inches='tight')
        plt.close()

    def plot_disease_simulations(self, model: str):
        output_dir = "./sens_data/epidemic_plot"
        os.makedirs(output_dir, exist_ok=True)
        if model == "rost":
            index = ["All contact", "0-4", "5-9", "10-14", "15-19", "20-24",
                     "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                     "55-59", "60-64", "65-69", "70-74", "75+"]

        elif model == "moghadas":
            index = ["All contact", "0-19", "20-49","50-65",  "65+"]
        elif model == "chikina":
            index = ["All contact", "0-4", "5-9", "10-14", "15-19", "20-24",
                     "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                     "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
        elif model == "seir":
            index = ["All contact", "0-4", "5-9", "10-14", "15-19", "20-24",
                     "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                     "55-59", "60-64", "65-69", "70+"]

        labels = ["$\sigma=0.5$, $\overline{\mathcal{R}}_0=1.2, {Ratio=0.75}$",
                  "$\sigma=0.5$, $\overline{\mathcal{R}}_0=1.2, {Ratio=0.5}$",
                  "$\sigma=0.5$, $\overline{\mathcal{R}}_0=1.8, {Ratio=0.75}$",
                  "$\sigma=0.5$, $\overline{\mathcal{R}}_0=1.8, {Ratio=0.5}$",
                  "$\sigma=0.5$, $\overline{\mathcal{R}}_0=2.5, {Ratio=0.75}$",
                  "$\sigma=0.5$, $\overline{\mathcal{R}}_0=2.5, {Ratio=0.5}$",
        "$\sigma=1.0$, $\overline{\mathcal{R}}_0=1.2, {Ratio=0.75}$",
                  "$\sigma=1.0$, $\overline{\mathcal{R}}_0=1.2, {Ratio=0.5}$",
        "$\sigma=1.0$, $\overline{\mathcal{R}}_0=1.8, {Ratio=0.75}$",
                  "$\sigma=1.0$, $\overline{\mathcal{R}}_0=1.8, {Ratio=0.5}$",
        "$\sigma=1.0$, $\overline{\mathcal{R}}_0=2.5, {Ratio=0.75}$",
                  "$\sigma=1.0$, $\overline{\mathcal{R}}_0=2.5, {Ratio=0.5}$"
                  ]

        df = pd.DataFrame(self.data.epidemic_data.T)
        df.index = labels
        df.columns = index
        # Reset the index to remove it
        df_reset = df.reset_index(drop=True)
        df_reset_numeric = df.apply(pd.to_numeric, errors='coerce')

        # Format values in scientific notation
        df_reset_formatted = df_reset_numeric.applymap(lambda x: f'{x:,.2e}'
        if not pd.isna(x) else '')

        # Plot the spacing as a blank column
        sns.heatmap(np.zeros((df_reset_numeric.shape[0], 1)),
                    cbar=False, cmap='viridis', annot=False)

        plt.figure(figsize=(12, 8))
        # Insert a blank column after the first column
        # df_reset_numeric.insert(1, 'reduce contact', np.nan)

        heatmap = sns.heatmap(df_reset_numeric, cmap='viridis',
                              annot=True, fmt=".2e", linewidths=.5)
        # Draw a vertical line after the first column
        plt.axvline(x=1.0, color='white', linestyle="solid", linewidth=15)
        # Remove y-axis label for the inserted column
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

        # Rotate x-axis labels
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90,
                                horizontalalignment='right', fontsize=18,
                                fontweight='bold')
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0,
                                horizontalalignment='right', fontsize=12,
                                fontweight='bold')
        heatmap.set_title('Moghadas model ICU', fontsize=35, fontweight='bold')

        # Adjust layout to avoid clipping
        plt.tight_layout()
        plt.savefig('./sens_data/epidemic_plot/' + 'epidemic_plot.pdf', format="pdf",
                    bbox_inches='tight')
        plt.close()

        plt.show()

    def plot_epidemic_size(self, time, params, cm_list, legend_list,
                           title_part, ratio, model: str):
        # Define the directory name
        output_dir = "./sens_data/Epidemic_size"
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        if model == "rost":
            compartments = ["l1", "l2", "ip", "ia1", "ia2", "ia3",
                            "is1", "is2", "is3", "ia1", "ia2", "ia3",
                            "ih", "ic", "icr"]
        elif model == "moghadas":
            compartments = ["e", "i_n",
                            "q_n", "i_h", "q_h", "a_n",
                            "a_q", "h", "c"]
        elif model == "chikina":
            compartments = ["i", "cp", "c"]
        elif model == "seir":
            compartments = ["e", "i"]

        plt.style.use('seaborn-darkgrid')
        colors = plt.cm.viridis(np.linspace(0, 1, len(cm_list)))
        plt.gca().set_prop_cycle('color', colors)
        fig, axs = plt.subplots(17, 1, figsize=(10, 30), sharex=True)

        # Create a ScalarMappable for the color bar
        cmap = get_cmap('viridis_r')

        # Initialize an empty list to store results
        results_list = []
        susc = self.sim_state["susc"]
        base_r0 = self.sim_state["base_r0"]
        for cm, legend in zip(cm_list, legend_list):
            # Get solution for the current combination
            solution = self.model.get_solution(
                init_values=self.model.get_initial_values,
                t=time,
                parameters=self.params,
                cm=cm
            )
            # Initialize an array to store the summed values for each compartment
            summed_values = np.zeros(len(solution))
            # Iterate over compartments and sum values
            for comp in compartments:
                comp_values = np.sum(
                    solution[:, self.model.c_idx[comp] *
                                self.n_ag:(self.model.c_idx[comp] + 1) * self.n_ag],
                    axis=1
                )
                summed_values += comp_values
            # Save the results in a dictionary with metadata
            result_entry = {
                'susc': susc,
                'base_r0': base_r0,
                'ratio': ratio,
                'n_infecteds_scaled': summed_values
            }

            # Append the result_entry to the results_list
            results_list.append(result_entry)
            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(results_list)

            # Plot the epidemic size for the current combination
            fig, ax = plt.subplots(figsize=(10, 6))
            color = cmap(0.3)

            ax.plot(time, result_entry['n_infecteds_scaled'],
                    label=f'susc={susc}, 'f'r0={base_r0},', color=color)
            ax.fill_between(time, 0, result_entry['n_infecteds_scaled'],
                            color=color, alpha=0.8)

            legend_text = f'susc={susc}, r0={base_r0}'
            # Plot the legend
            ax.legend([TriangleHandler()], [legend_text], fontsize=15,
                      labelcolor="green")
            # Get y-axis limits
            y_min, y_max = ax.get_ylim()
            # Calculate patch height based on the axis limits
            patch_height = y_max - y_min
            # Create a rectangular box for the y-axis label on the right side
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
                x_start = 200
                adj = 10
                text_adj = 4
            rect = patches.Rectangle((x_start, y_min), adj, patch_height,
                                     color="grey",
                                     linewidth=1, alpha=0.8, edgecolor='black',  # Set the edge color for the border
                                     linestyle='solid')
            ax.add_patch(rect)
            ax.text(x_start + text_adj, y_min + 0.5 * patch_height, 'Epidemic size population',
                    rotation=90,
                    verticalalignment='center', horizontalalignment='center',
                    fontsize=12, color='black'
                    )

            # Save the figure in the specified directory
            # Set the border width of the plot to match the linewidth of the rectangle
            ax.spines['top'].set_linewidth(5)
            ax.spines['bottom'].set_linewidth(5)
            ax.spines['left'].set_linewidth(5)
            ax.spines['right'].set_linewidth(0)
            ax.set_xlabel("day", fontsize=18)

        output_path = os.path.join(output_dir,
                                   f"epidemic_size_plot_{susc}_{base_r0}_{ratio}.pdf")
        plt.savefig(output_path, format="pdf",
                    bbox_inches='tight', pad_inches=0.5)
        plt.close()

        # Sum across all time points for each combination of 'susc' and 'base_r0'
        df['summed_n_infecteds'] = df['n_infecteds_scaled'].apply(np.sum)

        # Extract only necessary columns
        summed_df = df[['susc', 'base_r0', 'ratio', 'summed_n_infecteds']].drop_duplicates()

        # Add an 'age_group' column to your DataFrame based on the index
        df['age_group'] = df.index % (self.contact_matrix.shape[0] + 1)

        # Pivot the DataFrame
        pivot_df = df.pivot_table(index=['susc', 'base_r0', 'ratio'], columns='age_group',
                                  values='summed_n_infecteds',
                                  aggfunc='first')

        # Reset the index
        pivot_df = pivot_df.reset_index()
        # Save the entire DataFrame to a CSV file
        fname = "_".join([str(susc), str(base_r0), f"ratio_{ratio}"])
        filename = "sens_data/Epidemic_size" + "/" + fname

        np.savetxt(fname=filename + ".csv", X=np.sum(pivot_df)[3:],
                   delimiter=";")

    def plot_peak_size_epidemic(self, time, params, cm_list, legend_list,
                                title_part, ratio, model: str):
        # Define the directory name
        output_dir = "./sens_data/Epidemic_peak"
        os.makedirs(output_dir, exist_ok=True)
        cmap = get_cmap('viridis_r')

        if model == "rost":
            compartments = ["l1", "l2", "ip", "ia1", "ia2", "ia3",
                            "is1", "is2", "is3", "ia1", "ia2", "ia3",
                            "ih", "ic", "icr"]
        elif model == "moghadas":
            compartments = ["e", "i_n",
                            "q_n", "i_h", "q_h", "a_n",
                            "a_q", "h", "c"]
        elif model == "chikina":
            compartments = ["i", "cp", "c"]
        elif model == "seir":
            compartments = ["e", "i"]

        # Create a ScalarMappable for the color bar
        cmap = get_cmap('viridis_r')

        # Initialize an empty list to store results
        results_list = []
        max_peak_size_per_compartment = []
        for cm, legend in zip(cm_list, legend_list):
            # Get solution for the current combination
            solution = self.model.get_solution(
                init_values=self.model.get_initial_values,
                t=time,
                parameters=self.params,
                cm=cm
            )
            # Iterate over compartments and sum values
            for compartment in compartments:
                max_peak_size = 0
                for t_idx in range(len(solution)):
                    compartment_values = self.model.aggregate_by_age(
                        solution=solution,
                        idx=self.model.c_idx[compartment])
                    peak_size_at_t = compartment_values.sum()
                    max_peak_size = max(max_peak_size, peak_size_at_t)
                max_peak_size_per_compartment.append(max_peak_size)

                # Calculate the combined epidemic curve
                combined_curve = np.sum(
                    [solution[:,
                     self.model.c_idx[comp] * self.n_ag:(self.model.c_idx[comp] + 1) *
                                                        self.n_ag].sum(
                        axis=1)
                        for comp in compartments], axis=0)

            # Save the results in a dictionary with metadata
            result_entry = {
                'susc': self.sim_state['susc'],
                'base_r0': self.sim_state['base_r0'],
                'ratio': ratio,
                'n_infecteds_scaled': combined_curve
            }

            susc = self.sim_state['susc']
            base_r0 = self.sim_state['base_r0']
            # Append the result_entry to the results_list
            results_list.append(result_entry)
            # Plot the epidemic size for the current combination

            fig, ax = plt.subplots(figsize=(10, 6))
            color = cmap(0.3)
            ax.plot(time, result_entry['n_infecteds_scaled'],
                    label=f'susc={susc}, 'f'r0={base_r0},', color=color)
            ax.fill_between(time, 0, result_entry['n_infecteds_scaled'],
                            color=color, alpha=0.8)
            ax.legend([TriangleHandler()], [f'susc={susc}, r0={base_r0}'],
                      fontsize=15, labelcolor="green")

            # Find the peak value and its corresponding day
        peak_day = np.argmax(combined_curve)
        peak_value = combined_curve[peak_day]
        legend_text = f'susc={susc}, r0={base_r0},' \
                      f' Peak Size: {peak_value:.2f}'
        # Plot the legend
        ax.legend([TriangleHandler()], [legend_text], fontsize=15,
                  labelcolor="green")
        # Get y-axis limits
        y_min, y_max = ax.get_ylim()

        # Calculate patch height based on the axis limits
        patch_height = y_max - y_min

        # Create a rectangular box for the y-axis label on the right side
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
            adj = 100
            text_adj = 60
        elif model == "seir":
            x_start = 400
            adj = 50
            text_adj = 10
        rect = patches.Rectangle((x_start, y_min), adj, patch_height,
                                 color="grey",
                                 linewidth=4, alpha=0.8, edgecolor='black',  # Set the edge color for the border
                                 linestyle='solid')
        ax.add_patch(rect)
        ax.text(x_start + text_adj, y_min + 0.5 * patch_height,
                'Epidemic peak population',
                rotation=90,
                verticalalignment='center', horizontalalignment='center',
                fontsize=12, color='black'
                )
        # Save the figure in the specified directory
        # Set the border width of the plot to match the linewidth of the rectangle
        ax.spines['top'].set_linewidth(5)
        ax.spines['bottom'].set_linewidth(5)
        ax.spines['left'].set_linewidth(5)
        ax.spines['right'].set_linewidth(5)
        ax.set_xlabel("day", fontsize=18)
        # Set border color of the figure
        ax.patch.set_edgecolor('blue')
        output_path = os.path.join(output_dir,
                                   f"peak_size_plot_{susc}_{base_r0}_{ratio}.pdf")
        plt.savefig(output_path, format="pdf",
                    bbox_inches='tight', pad_inches=0.5)
        plt.close()
        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(results_list)
        # Sum across all time points for each combination of 'susc' and 'base_r0'
        df['summed_n_infecteds'] = df['n_infecteds_scaled'].apply(np.sum)
        # Extract only necessary columns
        summed_df = df[['susc', 'base_r0', 'ratio',
                        'summed_n_infecteds']].drop_duplicates()
        # Add an 'age_group' column to your DataFrame based on the index
        df['age_group'] = df.index % (self.contact_matrix.shape[0] + 1)
        # Pivot the DataFrame
        pivot_df = df.pivot_table(index=['susc', 'base_r0', 'ratio'], columns='age_group',
                                  values='summed_n_infecteds',
                                  aggfunc='first')
        # Reset the index
        pivot_df = pivot_df.reset_index()
        # Save the entire DataFrame to a CSV file
        fname = "_".join([str(susc), str(base_r0), f"ratio_{ratio}"])
        filename = "sens_data/Epidemic_peak" + "/" + fname
        np.savetxt(fname=filename + ".csv", X=np.sum(pivot_df)[3:],
                   delimiter=";")

    def plot_solution_final_death_size(self, time, params, cm_list, legend_list,
                                       title_part, ratio, model):
        output_dir = "./sens_data/death_size"
        os.makedirs(output_dir, exist_ok=True)
        cmap = get_cmap('viridis_r')

        results_list = []
        susc = self.sim_state["susc"]
        base_r0 = self.sim_state["base_r0"]
        for cm, legend in zip(cm_list, legend_list):
            solution = self.model.get_solution(
                init_values=self.model.get_initial_values,
                t=time,
                parameters=self.params,
                cm=cm)
            deaths = self.model.aggregate_by_age(
                solution=solution,
                idx=self.model.c_idx["d"])

            # Save the results in a dictionary with metadata
            result_entry = {
                'susc': susc,
                'base_r0': base_r0,
                'ratio': ratio,
                'n_deaths_scaled': deaths
            }

            # Append the result_entry to the results_list
            results_list.append(result_entry)

            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(results_list)

            # Plot the epidemic size for the current combination
            fig, ax = plt.subplots(figsize=(10, 6))
            color = cmap(0.3)
            ax.plot(time, result_entry['n_deaths_scaled'],
                    label=f'susc={susc}, 'f'r0={base_r0},', color=color)
            ax.fill_between(time, result_entry['n_deaths_scaled'],
                            color=color, alpha=0.8)

            # Plot the legend
            legend_text = f'susc={susc}, r0={base_r0}'
            ax.legend([TriangleHandler()], [legend_text], fontsize=15,
                      labelcolor="green")
            # Get y-axis limits
            y_min, y_max = ax.get_ylim()

            # Calculate patch height based on the axis limits
            patch_height = y_max - y_min

            # Create a rectangular box for the y-axis label on the right side
            if model == "rost":  # t = 1200
                x_start = 1200
                adj = 200
                text_adj = 35
            elif model == "moghadas":  # t = 400
                x_start = 400
                adj = 50
                text_adj = 10

            rect = patches.Rectangle((x_start, y_min), adj, patch_height,
                                     color="grey",
                                     linewidth=4, alpha=0.8, edgecolor='black',  # Set the edge color for the border
                                     linestyle='solid')
            ax.add_patch(rect)
            ax.text(x_start + text_adj, y_min + 0.5 * patch_height, 'deaths population',
                    rotation=90,
                    verticalalignment='center', horizontalalignment='center',
                    fontsize=12, color='black'
                    )
            # Set the border width of the plot to match the linewidth of the rectangle
            ax.spines['top'].set_linewidth(5)
            ax.spines['bottom'].set_linewidth(5)
            ax.spines['left'].set_linewidth(5)
            ax.spines['right'].set_linewidth(5)
            ax.set_xlabel("day", fontsize=18)
        output_path = os.path.join(output_dir,
                                   f"death_size_plot_{susc}_{base_r0}_{ratio}.pdf")
        plt.savefig(output_path, format="pdf",
                    bbox_inches='tight', pad_inches=0.5)
        plt.savefig(output_path, format="pdf")
        plt.close()

        # Sum across all time points for each combination of 'susc' and 'base_r0'
        df['summed_n_deaths'] = df['n_deaths_scaled'].apply(np.sum)
        # Extract only necessary columns
        summed_df = df[['susc', 'base_r0', 'ratio', 'summed_n_deaths']].drop_duplicates()

        # Add an 'age_group' column to your DataFrame based on the index
        df['age_group'] = df.index % (self.contact_matrix.shape[0] + 1)

        # Pivot the DataFrame
        pivot_df = df.pivot_table(index=['susc', 'base_r0', 'ratio'], columns='age_group',
                                  values='summed_n_deaths',
                                  aggfunc='first')
        # Reset the index
        pivot_df = pivot_df.reset_index()

        # Save the entire DataFrame to a CSV file
        fname = "_".join([str(susc), str(base_r0), f"ratio_{ratio}"])
        filename = "sens_data/death_size" + "/" + fname
        np.savetxt(fname=filename + ".csv", X=np.sum(pivot_df)[3:], delimiter=";")

    def plot_solution_hospitalized_peak_size(self, time, params, cm_list,
                                             legend_list, title_part, ratio, model):
        output_dir = "./sens_data/hospital"
        os.makedirs(output_dir, exist_ok=True)
        cmap = get_cmap('viridis_r')

        results_list = []
        susc = self.sim_state["susc"]
        base_r0 = self.sim_state["base_r0"]
        for cm, legend in zip(cm_list, legend_list):
            solution = self.model.get_solution(
                init_values=self.model.get_initial_values,
                t=time,
                parameters=self.params,
                cm=cm)

            if model == "chikina":
                total_hospitals = self.model.aggregate_by_age(
                    solution=solution,
                    idx=self.model.c_idx["cp"]) + self.model.aggregate_by_age(
                    solution=solution,
                    idx=self.model.c_idx["c"])
            elif model == "rost":
                total_hospitals = self.model.aggregate_by_age(
                    solution=solution, idx=self.model.c_idx["ih"]) + \
                                  self.model.aggregate_by_age(solution=solution,
                                                              idx=self.model.c_idx["ic"])
            elif model == "moghadas":
                total_hospitals = self.model.aggregate_by_age(
                    solution=solution,
                    idx=self.model.c_idx["i_h"]) + self.model.aggregate_by_age(
                    solution=solution,
                    idx=self.model.c_idx["q_h"]) + self.model.aggregate_by_age(
                    solution=solution,
                    idx=self.model.c_idx["h"])
            # Save the results in a dictionary with metadata
            result_entry = {
                'susc': susc,
                'base_r0': base_r0,
                'ratio': ratio,
                'n_hospitals_scaled': total_hospitals
            }
            # Append the result_entry to the results_list
            results_list.append(result_entry)
            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(results_list)

            # Plot the epidemic size for the current combination
            fig, ax = plt.subplots(figsize=(10, 6))
            color = cmap(0.3)
            ax.plot(time, result_entry['n_hospitals_scaled'], color=color)
            ax.fill_between(time, result_entry['n_hospitals_scaled'],
                            color=color, alpha=0.8)

            # Plot the legend
            legend_text = f'susc={susc}, r0={base_r0}'
            ax.legend([TriangleHandler()], [legend_text], fontsize=15,
                      labelcolor="green")
            # Get y-axis limits
            y_min, y_max = ax.get_ylim()

            # Calculate patch height based on the axis limits
            patch_height = y_max - y_min

            # Create a rectangular box for the y-axis label on the right side
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
                adj = 100
                text_adj = 60

            rect = patches.Rectangle((x_start, y_min), adj, patch_height,
                                     color="grey",
                                     linewidth=4, alpha=0.8, edgecolor='black',
                                     linestyle='solid')
            ax.add_patch(rect)
            ax.text(x_start + text_adj, y_min + 0.5 * patch_height,
                    'hospitalized population',
                    rotation=90,
                    verticalalignment='center', horizontalalignment='center',
                    fontsize=12, color='black'
                    )
            # Set the border width of the plot to match the linewidth of the rectangle
            ax.spines['top'].set_linewidth(5)
            ax.spines['bottom'].set_linewidth(5)
            ax.spines['left'].set_linewidth(5)
            ax.spines['right'].set_linewidth(5)
            ax.set_xlabel("day", fontsize=18)
        output_path = os.path.join(output_dir,
                                   f"hosp_size_plot_{susc}_{base_r0}_{ratio}.pdf")
        plt.savefig(output_path, format="pdf",
                    bbox_inches='tight', pad_inches=0.5)
        plt.savefig(output_path, format="pdf")
        plt.close()
        # Sum across all time points for each combination of 'susc' and 'base_r0'
        df['summed_n_hospital'] = df['n_hospitals_scaled'].apply(np.sum)
        # Extract only necessary columns
        summed_df = df[['susc', 'base_r0', 'ratio',
                        'summed_n_hospital']].drop_duplicates()

        # Add an 'age_group' column to your DataFrame based on the index
        df['age_group'] = df.index % (self.contact_matrix.shape[0] + 1)

        # Pivot the DataFrame
        pivot_df = df.pivot_table(index=['susc', 'base_r0', 'ratio'], columns='age_group',
                                  values='summed_n_hospital',
                                  aggfunc='first')
        # Reset the index
        pivot_df = pivot_df.reset_index()

        # Save the entire DataFrame to a CSV file
        fname = "_".join([str(susc), str(base_r0), f"ratio_{ratio}"])
        filename = "sens_data/hospital" + "/" + fname
        np.savetxt(fname=filename + ".csv", X=np.sum(pivot_df)[3:], delimiter=";")

    def plot_icu_size(self, time, params, susc, base_r0, cm_list,
                      legend_list, title_part, ratio, model):
        output_dir = "./sens_data/icu"
        os.makedirs(output_dir, exist_ok=True)
        cmap = get_cmap('viridis_r')

        susc = self.sim_state["susc"]
        base_r0 = self.sim_state["base_r0"]
        results_list = []
        for cm, legend in zip(cm_list, legend_list):
            solution = self.model.get_solution(
                init_values=self.model.get_initial_values,
                t=time,
                parameters=self.params,
                cm=cm)
            if model == "chikina":
                icu_values = self.model.aggregate_by_age(
                    solution=solution, idx=self.model.c_idx["c"])
            elif model == "rost":
                icu_values = self.model.aggregate_by_age(solution=solution,
                                                         idx=self.model.c_idx["ic"])
            elif model == "moghadas":
                icu_values = self.model.aggregate_by_age(
                    solution=solution, idx=self.model.c_idx["c"])
            # Save the results in a dictionary with metadata
            result_entry = {
                'susc': susc,
                'base_r0': base_r0,
                'ratio': ratio,
                'n_icu_scaled': icu_values
            }

            # Append the result_entry to the results_list
            results_list.append(result_entry)
            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(results_list)

            # Plot the epidemic size for the current combination
            fig, ax = plt.subplots(figsize=(10, 6))
            color = cmap(0.3)
            ax.plot(time, result_entry['n_icu_scaled'], color=color)
            ax.fill_between(time, result_entry['n_icu_scaled'],
                            color=color, alpha=0.8)

            # Plot the legend
            legend_text = f'susc={susc}, r0={base_r0}'
            ax.legend([TriangleHandler()], [legend_text], fontsize=15,
                      labelcolor="green")
            # Get y-axis limits
            y_min, y_max = ax.get_ylim()

            # Calculate patch height based on the axis limits
            patch_height = y_max - y_min

            # Create a rectangular box for the y-axis label on the right side
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
                adj = 100
                text_adj = 60

            rect = patches.Rectangle((x_start, y_min), adj, patch_height,
                                     color="grey",
                                     linewidth=4, alpha=0.8, edgecolor='black',
                                     linestyle='solid')
            ax.add_patch(rect)
            ax.text(x_start + text_adj, y_min + 0.5 * patch_height,
                    'ICU population',
                    rotation=90,
                    verticalalignment='center', horizontalalignment='center',
                    fontsize=12, color='black'
                    )
            # Set the border width of the plot to match the linewidth of the rectangle
            ax.spines['top'].set_linewidth(5)
            ax.spines['bottom'].set_linewidth(5)
            ax.spines['left'].set_linewidth(5)
            ax.spines['right'].set_linewidth(5)
            ax.set_xlabel("day", fontsize=18)
        output_path = os.path.join(output_dir,
                                   f"ICU_size_plot_{susc}_{base_r0}_{ratio}.pdf")
        plt.savefig(output_path, format="pdf",
                    bbox_inches='tight', pad_inches=0.5)
        plt.savefig(output_path, format="pdf")
        plt.close()

        # Sum across all time points for each combination of 'susc' and 'base_r0'
        df['summed_n_icu'] = df['n_icu_scaled'].apply(np.sum)
        # Extract only necessary columns
        summed_df = df[['susc', 'base_r0', 'ratio', 'summed_n_icu']].drop_duplicates()

        # Add an 'age_group' column to your DataFrame based on the index
        df['age_group'] = df.index % (self.contact_matrix.shape[0] + 1)

        # Pivot the DataFrame
        pivot_df = df.pivot_table(index=['susc', 'base_r0', 'ratio'], columns='age_group',
                                  values='summed_n_icu',
                                  aggfunc='first')
        # Reset the index
        pivot_df = pivot_df.reset_index()

        # Save the entire DataFrame to a CSV file
        fname = "_".join([str(susc), str(base_r0), f"ratio_{ratio}"])
        filename = os.path.join("sens_data/icu", fname + ".csv")
        np.savetxt(fname=filename, X=np.sum(pivot_df)[3:], delimiter=";")

class TriangleHandler(Line2D):
    def __init__(self, *args, **kwargs):
        # Default line properties
        line_props = {'linewidth': 0, 'linestyle': '-'}

        # Extract label and color from kwargs
        label = kwargs.pop('label', None)
        color = kwargs.pop('color', 'green')

        # Call the parent constructor
        super().__init__([], [], label=label, color=color, marker='^',
                         markersize=12, **line_props)

