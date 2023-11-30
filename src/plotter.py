import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import LogLocator, LogFormatterSciNotation as LogFormatter
import matplotlib.colors as colors
import matplotlib.collections as collections
from cycler import cycler

from matplotlib.tri import Triangulation
import numpy as np
import pandas as pd
import seaborn as sns

from src.dataloader import DataLoader
from src.simulation_base import SimulationBase
from src.prcc import get_rectangular_matrix_from_upper_triu

plt.style.use('seaborn-whitegrid')


class Plotter:
    def __init__(self, sim_obj: SimulationBase) -> None:

        self.deaths = None
        self.hospitalized = None

        self.data = DataLoader()
        self.sim_obj = sim_obj
        self.f1 = np.array([])
        self.f2 = np.array([])
        self.f3 = np.array([])
        self.f4 = np.array([])
        self.f5 = np.array([])
        self.f6 = np.array([])
        self.contact_full = np.array([])

    def plot_contact_matrices_hungary(self, filename):
        os.makedirs("sens_data/contact_matrices", exist_ok=True)
        cmaps = {"Home": "Greens", "Work": "Greens", "School": "Greens",
                 "Other": "Greens", "Full": "Greens"}
        contact_full = np.array([self.data.contact_data[i]
                                 for i in list(cmaps.keys())[:-1]]).sum(axis=0)
        self.contact_full = contact_full
        for i in cmaps.keys():
            contacts = self.data.contact_data[i] if i != "Full" else contact_full
            param_list = range(0, 16, 1)
            contact_matrix = pd.DataFrame(contacts, columns=param_list, index=param_list)
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif', size=14)
            plot = plt.imshow(contact_matrix, cmap=cmaps[i], origin='lower',
                              alpha=.9, interpolation="nearest", vmin=0, vmax=5)
            number_of_age_groups = 16
            plt.gca().set_xticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
            plt.gca().set_yticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
            plt.gca().grid(which='minor', color='gray', linestyle='-', linewidth=1)
            plt.xticks(ticks=param_list, rotation=0, fontsize=15)
            plt.yticks(ticks=param_list, fontsize=15)
            if i == "Full":
                cbar = plt.colorbar(plot, pad=0.02, fraction=0.04)
                tick_font_size = 40
                cbar.ax.tick_params(labelsize=tick_font_size)
            plt.title(i + " contact", y=1.03, fontsize=25)
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
        cmaps = ["Greens", p_value_cmap]

        log_norm = colors.LogNorm(vmin=1e-3, vmax=1e0)  # used for p_values
        norm = plt.Normalize(vmin=0, vmax=1)  # used for PRCC_values

        fig, ax = plt.subplots()
        triang = self.construct_triangle_grids_prcc_p_value(prcc_vector=prcc_vector,
                                                            p_values=p_values)
        mask = self.get_mask_and_values(prcc_vector=prcc_vector, p_values=p_values)
        images = [ax.tripcolor(t, np.ravel(val), cmap=cmap, ec="white")
                  for t, val, cmap in zip(triang,
                                          mask, cmaps)]

        fig.colorbar(images[0], ax=ax, shrink=0.7, aspect=20 * 0.7)  # for the prcc values
        cbar_pval = fig.colorbar(images[1], ax=ax, shrink=0.7, aspect=20 * 0.7, pad=0.1)

        images[1].set_norm(norm=log_norm)
        images[0].set_norm(norm=norm)

        locator = LogLocator()
        formatter = LogFormatter()
        cbar_pval.locator = locator
        cbar_pval.formatter = formatter
        cbar_pval.update_normal(images[1])

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
    def aggregated_prcc_pvalues_plots(param_list, prcc_vector, p_values,
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
        for pos, y, err in zip(xp, list(abs(prcc_vector)), list(abs(p_values))):
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

    def plot_aggregation_prcc_pvalues(self, prcc_vector, p_values, filename_without_ext):
        title_list = filename_without_ext.split("_")
        plot_title = '$\overline{\mathcal{R}}_0=$' + title_list[1]
        self.aggregated_prcc_pvalues_plots(param_list=16,
                                           prcc_vector=prcc_vector, p_values=p_values,
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
        col.set_array(a.flatten())
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
        ax.set_xticks(range(self.sim_obj.n_ag), labels=self.deaths.columns, rotation=90,
                      fontsize=20)
        ax.set_yticks(range(6), labels=self.deaths.index, fontsize=20)
        fig.colorbar(im1, ax=ax, shrink=0.7, aspect=20 * 0.7, pad=0.01)
        plt.savefig('./sens_data/hosp_death/' + 'hosp_death.pdf', format="pdf",
                    bbox_inches='tight')
        plt.close()

    def plot_solution_final_death_size(self, time, params, cm_list, legend_list, title_part):
        os.makedirs("./sens_data/dynamics", exist_ok=True)
        plt.rcParams['axes.prop_cycle'] = cycler('color', plt.get_cmap('tab20').colors)
        fig = plt.figure(figsize=(10, 10))
        for idx, cm in enumerate(cm_list):
            solution = self.model.get_solution(
                init_values=self.model.get_initial_values,
                t=time,
                parameters=self.params,
                cm=cm)
            plt.plot(time, np.sum(solution[:, self.model.c_idx["d"] *
                                  self.n_ag:(self.model.c_idx["d"] + 1) * self.n_ag], axis=1),
                     label=legend_list[idx])
        plt.legend()
        plt.gca().set_xlabel('days')
        plt.gca().set_ylabel('age deaths')
        plt.gca().set_xlim([0, 800])
        plt.gca().set_ylim([1000, 40000])
        plt.tight_layout()
        plt.savefig('./sens_data/dynamics' + "/" + title_part + '.pdf', format="pdf")
        plt.close()

    def plot_solution_hospitalized_peak_size(self, time, params, cm_list, legend_list, title_part):
        os.makedirs("./sens_data/dynamics", exist_ok=True)
        plt.rcParams['axes.prop_cycle'] = cycler('color', plt.get_cmap('tab20').colors)
        for idx, cm in enumerate(cm_list):
            solution = self.model.get_solution(
                init_values=self.model.get_initial_values,
                t=time,
                parameters=self.params,
                cm=cm)
            plt.plot(time, np.max(solution[:, self.model.c_idx["ih"] *
                                              self.n_ag:(self.model.c_idx["ih"] + 1) * self.n_ag],
                                  axis=1), label=legend_list[idx])
        plt.legend()
        plt.gca().set_xlabel('days')
        plt.gca().set_ylabel('age hospitalized')
        plt.gca().set_xlim([0, 300])
        plt.gca().set_ylim([100, 20000])
        plt.gcf().set_size_inches(10, 10)
        plt.tight_layout()
        plt.savefig('./sens_data/dynamics' + "/" + title_part + '.pdf', format="pdf")
        plt.close()



