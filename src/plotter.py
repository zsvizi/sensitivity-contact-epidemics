import os
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
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
        self.prcc_vector = None

    def generate_axis_label(self):
        # Define parameters for analysis
        ccc = np.chararray((1, 16, 16))[0]
        ccc[:] = r"$"
        xxx = np.chararray((1, 16, 16))[0]
        xxx[:] = "X"
        vesz = np.chararray((1, 16, 16))[0]
        vesz[:] = ','
        alul = np.chararray((1, 16, 16))[0]
        alul[:] = '_'
        bal = np.chararray((1, 16, 16))[0]
        bal[:] = '{'
        jobb = np.chararray((1, 16, 16))[0]
        jobb[:] = '}'
        r = np.arange(16)
        vvv = np.repeat(np.reshape(r, (-1, 1)), 16, axis=1).astype(str)
        names = ccc.astype(str) + xxx.astype(str) + alul.astype(str) + bal.astype(str) + \
                vvv + vesz.astype(str) + vvv.T + jobb.astype(str) + ccc.astype(str)
        return names

    def plot_prcc_values_as_heatmap(self, prcc_vector, filename, filename_without_ext):
        os.makedirs("sens_data/PRCC_heatmap", exist_ok=True)
        number_of_age_groups = 16
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.margins(0, tight=False)
        cmap = mcolors.LinearSegmentedColormap.from_list("", [mcolors.CSS4_COLORS["lightgrey"],
                                                              mcolors.CSS4_COLORS["crimson"]])
        param_list = range(0, number_of_age_groups, 1)
        new_contact_mtx = np.zeros((number_of_age_groups, number_of_age_groups))
        new_contact_mtx[self.sim_obj.upper_tri_indexes] = prcc_vector
        new_2 = new_contact_mtx.T
        new_2[self.sim_obj.upper_tri_indexes] = prcc_vector
        corr = pd.DataFrame(new_2, columns=param_list, index=param_list)
        plt.figure(figsize=(12, 12))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        title_list = filename_without_ext.split("_")
        plt.title('PRCC results for ' + title_list[-1] + ' version with\nchildren susceptibility ' + title_list[1] +
                  'and base R0=' + title_list[2], y=1.03, fontsize=25)
        plt.imshow(corr, cmap=cmap, origin='lower')
        plt.colorbar(pad=0.02, fraction=0.04)
        # plt.grid(b=None)
        plt.gca().set_xticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
        plt.gca().set_yticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
        plt.gca().grid(which='minor', color='w', linestyle='-', linewidth=2)
        plt.xticks(ticks=param_list, labels=param_list)
        plt.yticks(ticks=param_list, labels=param_list)
        plt.savefig('./sens_data/PRCC_heatmap/' + filename + '.pdf', format="pdf", bbox_inches='tight')
        plt.close()

    def plot_prcc_values_lockdown_3(self, prcc_vector, filename_to_save, plot_title):
        os.makedirs("sens_data/PRCC_bars", exist_ok=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.margins(0, tight=False)
        parameter_count = len(list(prcc_vector)) // 3
        # parameter_count = 16
        age_group_vector = list(range(parameter_count)) * 3
        contact_type_vector = ["school"] * 16 + ["work"] * 16 + ["other"] * 16
        data_to_df = [[age_group_vector[i], contact_type_vector[i], prcc_vector[i]] for i in range(parameter_count * 3)]
        df = pd.DataFrame(data_to_df, columns=['Age group', 'Contact type', 'val'])
        plt.figure(figsize=(17, 10))
        colors = {"home": "#ff96da", "work": "#96daff", "school": "#96ffbb", "other": "#ffbb96", "total": "blue"}
        dftemp = df.pivot("Age group", "Contact type", "val")
        dftemp = dftemp[["school", "work", "other"]]
        dftemp.plot(kind='bar', width=0.85, color=[colors["school"], colors["work"], colors["other"]])
        xp = range(parameter_count)
        param_list = map(lambda x: "75+" if x == 15 else str(5 * x) + "-" + str(5 * x + 4), range(parameter_count))
        plt.tick_params(direction="in")
        plt.xticks(ticks=xp, labels=param_list, rotation=45)
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.ylabel('PRCC indices', labelpad=10, fontsize=20)
        plt.xlabel('Age groups', labelpad=10, fontsize=20)
        plt.title(plot_title, y=1.03, fontsize=25)
        plt.savefig('./sens_data/PRCC_bars/' + filename_to_save + '.pdf', format="pdf", bbox_inches='tight')
        plt.close()

    def plot_prcc_values(self, param_list, prcc_vector, filename_without_ext, filename_to_save, plot_title):
        os.makedirs("sens_data/PRCC_plot", exist_ok=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.margins(0, tight=False)
        parameter_count = len(list(prcc_vector))
        cmap = mcolors.LinearSegmentedColormap.from_list("", [mcolors.CSS4_COLORS["lightgrey"],
                                                              mcolors.CSS4_COLORS["crimson"]])
        xp = range(parameter_count)
        plt.figure(figsize=(35, 10))
        plt.tick_params(direction="in")
        # aggregation use abs values
        color_list = ["#ff96da", "#96ffbb", "#96daff", "#ffbb96"]
        # colors = [color_list[0] if i < xp[16] else color_list[1] if i < xp[32] else color_list[3] for i in xp]
        # plt.bar(xp, list(abs(prcc_vector)), align='center', color=cmap(prcc_vector))
        plt.bar(xp, list(abs(prcc_vector)), align='center', color=color_list[0])
        # x = plt.bar(xp, list(abs(prcc_vector)), align='center', color=colors)
        plt.xticks(ticks=xp, labels=param_list, rotation=90)
        plt.yticks(ticks=np.arange(-1, 3.7, 0.1))
        axes = plt.gca()
        plt.legend()
        # agg legend
        # plt.legend((x[0:16], x[16:32], x[32:48]), ('school', 'work', 'other'))
        axes.set_ylim([0, 3.7])
        plt.ylabel('Aggregated PRCC indices', labelpad=10, fontsize=20)
        plt.xlabel('Age groups', labelpad=10, fontsize=20)
        title_list = filename_without_ext.split("_")
        plt.title(plot_title, y=1.03, fontsize=25)
        plt.savefig('./sens_data/PRCC_plot/' + filename_to_save + '.pdf', format="pdf", bbox_inches='tight')
        plt.close()

    def stacked_prcc_pvalues(self, param_list, prcc_vector, p_values, filename_without_ext,
                             filename_to_save, plot_title):
        os.makedirs("sens_data/PRCC_PVAL_PLOT", exist_ok=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.margins(0, tight=False)
        # parameter_count = len(list(prcc_vector))
        parameter_count = 16
        xp = range(parameter_count)
        # plt.figure(figsize=(35, 10))
        plt.figure(figsize=(15, 12))
        color_list = ["#ff96da", "#96ffbb", "#96daff", "#ffbb96"]
        colors = ["r" if i < 0.01 else "orange" if 0.01 < i < 0.05 else "pink" if 0.05 < i < 0.1 else
        "purple" for i in p_values]
        plt.tick_params(direction="in")
        plt.bar(xp, list(prcc_vector), align='center', width=0.6, color="g", label="PRCC")  # used absolute prcc
        plt.bar(xp, list(p_values < 0.01), width=0.6, align='center', bottom=np.array(prcc_vector),
                  color="r", label="P_value less than 0.01")
        plt.bar(xp, list([0.01 < i < 0.05 for i in p_values]), align='center', width=0.6,
                  bottom=np.array(prcc_vector), color="orange", label="P_value 0.01 - 0.05")
        plt.bar(xp, list([0.05 < i < 0.1 for i in p_values]), align='center', width=0.6,
                  bottom=np.array(prcc_vector), color="pink", label="P_value 0.05 - 0.1")
        plt.bar(xp, list(p_values > 0.1), align='center', width=0.6, bottom=np.array(prcc_vector),
                  color="purple", label="P_value greater than 0.1")
        # plt.xticks(ticks=xp, labels=param_list, rotation=90)
        plt.xticks(ticks=xp, rotation=90)
        plt.yticks(ticks=np.arange(-1, 2.5, 0.2))
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0, 2.5])
        plt.ylabel('PRCC and P_values', labelpad=10, fontsize=20)
        plt.xlabel('Pairs of age groups', labelpad=10, fontsize=20)
        plt.title("PRCC values and their corresponding P values")
        title_list = filename_without_ext.split("_")
        plt.title(plot_title, y=1.03, fontsize=20)
        plt.savefig('./sens_data/PRCC_PVAL_PLOT/' + filename_to_save + '.pdf', format="pdf", bbox_inches='tight')
        plt.close()

    def plot_prcc_p_values_as_heatmap(self, prcc, p_values, filename_without_ext, filename_to_save, plot_title):
        os.makedirs("sens_data/heatmap", exist_ok=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.margins(0, tight=False)
        PRC_mtx = get_rectangular_matrix_from_upper_triu(prcc[:self.sim_obj.upper_tri_size], self.sim_obj.n_ag)
        P_values_mtx = get_rectangular_matrix_from_upper_triu(p_values[:self.sim_obj.upper_tri_size], self.sim_obj.n_ag)
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
        triangul = [Triangulation(x, y, triangles) for triangles in [trianglesPRCC, trianglesP]]
        values = PRC_mtx, P_values_mtx
        cmaps = ['Greens', 'Reds']
        norms = [plt.Normalize(-0.001, 1) for _ in range(2)]

        fig, ax = plt.subplots()
        images = [ax.tripcolor(t, np.ravel(val), cmap=cmap, norm=norm, ec='white')
                  for t, val, cmap, norm in zip(triangul, values, cmaps, norms)]

        cbar = fig.colorbar(images[0], ax=ax, shrink=0.7, aspect=20 * 0.7)
        cbar = fig.colorbar(images[1], ax=ax, shrink=0.7, aspect=20 * 0.7)
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

    def plot_aggregation(self, agg_values, filename_without_ext):
        title_list = filename_without_ext.split("_")
        plot_title = 'Target: R0, Susceptibility=' + title_list[0] + ', R0=' + title_list[1] + \
                     ', Aggregation: ' + title_list[3]
        self.plot_prcc_values(np.arange(16), agg_values, filename_without_ext, "agg_plot_" +
                              filename_without_ext, plot_title)

    def generate_prcc_plots(self, prcc_vector, p_values, filename_without_ext):
        title_list = filename_without_ext.split("_")
        names = np.array(self.generate_axis_label()[self.sim_obj.upper_tri_indexes])
        plot_title = 'Target: Epidemic size, Susceptibility=' + title_list[0] + ', R0=' + title_list[1]
        # self.plot_prcc_p_values_as_heatmap(prcc_vector, p_values, filename_without_ext,
        #                                     "PRCC_P_VALUES" + filename_without_ext + "_R0", plot_title)
        self.stacked_prcc_pvalues(16, prcc_vector, p_values, filename_without_ext,
                                  "PRCC_P_VALUES_" + filename_without_ext + "_R0", plot_title)
        # self.stacked_prcc_pvalues(names.flatten().tolist(), prcc_vector, p_values, filename_without_ext,
        #                            "PRCC_P_VALUES" + filename_without_ext + "_R0", plot_title)
        # self.plot_prcc_values(names.flatten().tolist(), prcc_vector, filename_without_ext,
        #                       "PRCC_plot_" + filename_without_ext + "_R0", plot_title)

    def generate_lockdown3_prcc(self, prcc_vector, filename_without_ext):
        title_list = filename_without_ext.split("_")
        plot_title = 'Target: R0, Susceptibility=' + title_list[0] + ', R0=' + title_list[1]
        self.plot_prcc_values_lockdown_3(prcc_vector, "PRCC_bars_" + filename_without_ext + "_R0", plot_title)
        # if "lockdown_3" not in filename_without_ext:

    def scatter_plot(self, p_values, prcc_vector):
        plt.figure(figsize=(16, 10))
        plt.plot(p_values, prcc_vector, linewidth=3.0, linestyle='--')
        plt.ylabel("PRCC", fontsize=20)
        plt.xlabel("time (days)", fontsize=20)
        plt.show()

    def plot_symm_contact_matrix_as_bars(self, param_list, contact_vector, file_name):
        ymax = max(contact_vector)
        color_map = mcm.get_cmap('Blues')
        my_norm = mcolors.Normalize(vmin=-ymax / 5, vmax=ymax)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.margins(0, tight=False)
        parameter_count = len(list(contact_vector))
        xp = range(parameter_count)
        plt.figure(figsize=(35, 10))
        plt.tick_params(direction="in")
        plt.bar(xp, list(contact_vector), align='center', color=color_map(my_norm(contact_vector)))
        plt.xticks(ticks=xp, labels=param_list, rotation=90)
        axes = plt.gca()
        axes.set_ylim([0, 1.1 * ymax])
        plt.ylabel('Number of contacts', labelpad=10, fontsize=20)
        plt.xlabel('Pairs of age groups', labelpad=10, fontsize=20)
        plt.title('Total contact matrix of Hungary', y=1.03, fontsize=25)
        plt.savefig("./sens_data/" + file_name + ".pdf", format="pdf", bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_stacked(list_of_flattened_matrices, param_list, labels, color_list, title=None, filename=None):
        ymax = max(np.array(list_of_flattened_matrices).sum(axis=0))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.margins(0, tight=False)
        parameter_count = len(list(param_list))
        xp = range(parameter_count)
        plt.figure(figsize=(35, 10))
        plt.tick_params(direction="in")
        plt.bar(xp, list(list_of_flattened_matrices[0]), align='center', label=labels[0], color=color_list[0])
        plt.bar(xp, list(list_of_flattened_matrices), align='center', label=param_list, color=color_list[0])
        for t in range(1, len(list_of_flattened_matrices)):
            plt.bar(xp, list(list_of_flattened_matrices[t]), align='center',
                    bottom=np.array(list_of_flattened_matrices[:t]).sum(axis=0), label=labels[t], color=color_list[t])
        plt.bar(xp, list(list_of_flattened_matrices), align='center',
                bottom=np.array(list_of_flattened_matrices).sum(axis=0), label=param_list, color=color_list[0])
        plt.xticks(ticks=xp, labels=param_list, rotation=90)
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0, 1.1 * ymax])
        plt.ylabel('Number of contacts', labelpad=10, fontsize=20)
        plt.xlabel('Pairs of age groups', labelpad=10, fontsize=20)
        if title:
            plt.title(title, y=1.03, fontsize=25)
        plt.savefig("./sens_data/Stacked_" + filename + ".pdf", format="pdf", bbox_inches='tight')
        plt.close()

    def plot_2d_contact_matrices(self):
        colors = {"home": "#ff96da", "work": "#96daff", "school": "#96ffbb", "other": "#ffbb96", "total": "blue"}
        cmaps = {"home": "Oranges", "work": "Reds", "school": "Greens", "other": "Blues", "total": "Purples"}
        contact_total = np.array([self.data.contact_data[t] for t in list(colors.keys())[:-1]]).sum(axis=0)
        for t in colors.keys():
            contact_small = self.data.contact_data[t] if t != "total" else contact_total
            param_list = range(0, 16, 1)
            corr = pd.DataFrame(contact_small * self.data.age_data, columns=param_list, index=param_list)
            plt.figure(figsize=(12, 12))
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif', size=14)
            plt.title('Symmetric ' + t + ' contact matrix of Hungary', y=1.03, fontsize=25)
            plt.imshow(corr, cmap=cmaps[t], origin='lower')
            plt.colorbar(pad=0.02, fraction=0.04)
            #plt.grid(b=None)
            number_of_age_groups = 16
            age_groups = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
                          "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75+"]

            plt.gca().set_xticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
            plt.gca().set_yticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
            plt.gca().grid(which='minor', color='gray', linestyle='-', linewidth=1)
            plt.xticks(ticks=param_list, labels=age_groups, rotation=90)
            plt.yticks(ticks=param_list, labels=age_groups)
            plt.savefig('./sens_data/CM_sym_Hungary' + "_" + t + '.pdf', format="pdf", bbox_inches='tight')
            plt.close()

    def generate_stacked_plots(self):
        sym_cm_total = self.sim_obj.contact_matrix * self.sim_obj.age_vector
        self.plot_symm_contact_matrix_as_bars(np.array(self.generate_axis_label()[
                                                           self.sim_obj.upper_tri_indexes]).flatten().tolist(),
                                              sym_cm_total[self.sim_obj.upper_tri_indexes],
                                              file_name="CM_sym_Hungary")

        # Plot stacked home-work-school-total
        c_matrices = []
        c_names = np.array(["home", "school", "work", "other"])
        color_list = ["#ff96da", "#96ffbb", "#96daff", "#ffbb96"]
        for n in c_names:
            contact_matrix = self.data.contact_data[n]
            sym_cm = contact_matrix * self.sim_obj.age_vector
            c_matrices.append(sym_cm[self.sim_obj.upper_tri_indexes])
        self.plot_stacked(c_matrices, np.array(self.generate_axis_label()[
                                                   self.sim_obj.upper_tri_indexes]).flatten().tolist(), c_names,
                          color_list=color_list, title="Stacked version of different types of contacts",
                          filename="HWSO_contacts")

        # Plot stacked total-normed-home
        c_matrices = []
        c_names = ["home", "normed", "total"]
        color_list = ["#ff96da", "#ecff7f", "#56a8ff"]
        sym_cm_1 = self.data.contact_data["home"] * self.sim_obj.age_vector
        sym_cm_2 = sym_cm_total - np.min(sym_cm_total - sym_cm_1)
        c_matrices.append(sym_cm_1[self.sim_obj.upper_tri_indexes])
        c_matrices.append(sym_cm_2[self.sim_obj.upper_tri_indexes] - sym_cm_1[self.sim_obj.upper_tri_indexes])
        c_matrices.append(sym_cm_total[self.sim_obj.upper_tri_indexes] - sym_cm_2[self.sim_obj.upper_tri_indexes])
        self.plot_stacked(c_matrices, np.array(self.generate_axis_label()[
                                                   self.sim_obj.upper_tri_indexes]).flatten().tolist(), c_names,
                          color_list=color_list, title="Compound version of total, normed and home contacts",
                          filename="TNH_contacts")

    def plot_contact_matrix_as_grouped_bars(self):
        os.makedirs("./sens_data", exist_ok=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.margins(0, tight=False)
        parameter_count = 16
        xp = range(parameter_count)
        param_list = list(
            map(lambda x: "75+" if x == 15 else str(5 * x) + "-" + str(5 * x + 4), range(parameter_count)))

        age_group_vector = list(range(parameter_count)) * 3
        contact_type_vector = ["school"] * 16 + ["work"] * 16 + ["other"] * 16

        colors = {"home": "#ff96da", "work": "#96daff", "school": "#96ffbb", "other": "#ffbb96", "total": "blue"}

        contact_school = np.sum(self.data.contact_data["school"] * self.sim_obj.age_vector, axis=1)
        contact_work = np.sum(self.data.contact_data["work"] * self.sim_obj.age_vector, axis=1)
        contact_other = np.sum(self.data.contact_data["other"] * self.sim_obj.age_vector, axis=1)
        contacts_as_vector = np.array([contact_school, contact_work, contact_other]).flatten()

        plt.figure(figsize=(17, 10))
        data_to_df = [[age_group_vector[i], contact_type_vector[i], contacts_as_vector[i]] for i in
                      range(parameter_count * 3)]
        df = pd.DataFrame(data_to_df, columns=['Age group', 'Contact type', 'val'])
        dftemp = df.pivot("Age group", "Contact type", "val")
        dftemp = dftemp[["school", "work", "other"]]
        dftemp.plot(kind='bar', width=0.85, color=[colors["school"], colors["work"], colors["other"]])

        plt.tick_params(direction="in")
        plt.xticks(ticks=xp, labels=param_list, rotation=45)
        plt.xlabel('Age groups', labelpad=10, fontsize=20)
        plt.title("Contacts from symmetric CMs", y=1.03, fontsize=25)
        plt.savefig('./sens_data/CM_sym_grouped.pdf', format="pdf", bbox_inches='tight')
        plt.close()

        contact_school_2 = np.sum(self.data.contact_data["school"], axis=1)
        contact_work_2 = np.sum(self.data.contact_data["work"], axis=1)
        contact_other_2 = np.sum(self.data.contact_data["other"], axis=1)
        contacts_vector_2 = np.array([contact_school_2, contact_work_2, contact_other_2]).flatten()
        data_df = [[age_group_vector[i], contact_type_vector[i], contacts_vector_2[i]] for i in
                   range(parameter_count * 3)]
        df_2 = pd.DataFrame(data_df, columns=['Age group', 'Contact type', 'val'])
        dftemp_2 = df_2.pivot("Age group", "Contact type", "val")
        dftemp_2 = dftemp_2[["school", "work", "other"]]
        dftemp_2.plot(kind='bar', width=0.85, color=[colors["school"], colors["work"], colors["other"]])

        plt.tick_params(direction="in")
        plt.xticks(ticks=xp, labels=param_list, rotation=45)
        plt.xlabel('Age groups', labelpad=10, fontsize=20)
        plt.title("Contacts from original CMs", y=1.03, fontsize=25)
        plt.savefig('./sens_data/CM_prem_grouped.pdf', format="pdf", bbox_inches='tight')
        plt.close()

    def plot_model(self, params, cm, filename_without_ext, filename_to_save, plot_title):
        os.makedirs("sens_data/death", exist_ok=True)
        cm_list = []
        time = np.arange(0, 250, 0.5)
        color_list = ["#ff96da", "#96ffbb", "#96daff", "#ffbb96"]
        # plot uses contact matrix
        solution_cmx = self.sim_obj.model.get_solution(t=time, parameters=params, cm=self.sim_obj.contact_matrix)
        for idx, cm in enumerate(cm_list):
            # plot uses manipulated contact matrix (cm)
            solution = self.sim_obj.model.get_solution(t=time, parameters=params, cm=cm)
            # consider the total number of deaths
            plt.plot(time, np.sum(solution[:, self.sim_obj.model.c_idx["d"] *
                                              self.sim_obj.n_ag:(self.sim_obj.model.c_idx["d"] + 1) *
                                                      self.sim_obj.n_ag], axis=1),
                     label="all ages", color=color_list[0], linewidth=3.0)
            plt.plot(time, np.sum(solution_cmx[:, self.sim_obj.model.c_idx["d"] *
                                                  self.sim_obj.n_ag:(self.sim_obj.model.c_idx["d"] + 1) *
                                                                self.sim_obj.n_ag], axis=1),
                     label="all ages c_mtx", color=color_list[0], linewidth=3.0, linestyle='--')
            # 75+
            plt.plot(time, np.sum(solution[:, self.sim_obj.model.c_idx["d"] *
                                              self.sim_obj.n_ag + 15:self.sim_obj.model.c_idx["d"] *
                                                                     self.sim_obj.n_ag + 16], axis=1),
                     label="75+", color="blue", linewidth=3.0)
            plt.plot(time, np.sum(solution_cmx[:, self.sim_obj.model.c_idx["d"] *
                                                  self.sim_obj.n_ag + 15:self.sim_obj.model.c_idx["d"] *
                                                                     self.sim_obj.n_ag + 16], axis=1),
                     label="75+ c_mtx", color="blue", linewidth=3.0, linestyle='--')

            # age 65-75
            plt.plot(time, np.sum(solution[:, self.sim_obj.model.c_idx["d"] *
                                              self.sim_obj.n_ag + 13:self.sim_obj.model.c_idx["d"] *
                                                           self.sim_obj.n_ag + 15], axis=1),
                     label="65-74", color=color_list[1], linewidth=3.0)
            plt.plot(time, np.sum(solution_cmx[:, self.sim_obj.model.c_idx["d"] *
                                                  self.sim_obj.n_ag + 13:self.sim_obj.model.c_idx["d"] *
                                                                     self.sim_obj.n_ag + 15], axis=1),
                     label="65-74 c_mtx", color=color_list[1], linewidth=3.0, linestyle='--')
            # age 25-64
            plt.plot(time, np.sum(solution[:, self.sim_obj.model.c_idx["d"] *
                                              self.sim_obj.n_ag + 5:self.sim_obj.model.c_idx["d"] *
                                                           self.sim_obj.n_ag + 13], axis=1),
                     label="25-64", color="orange", linewidth=3.0)
            plt.plot(time, np.sum(solution_cmx[:, self.sim_obj.model.c_idx["d"] *
                                                  self.sim_obj.n_ag + 5:self.sim_obj.model.c_idx["d"] *
                                                                    self.sim_obj.n_ag + 13], axis=1),
                     label="25-64", color="orange", linewidth=3.0, linestyle='--')
            # age 0-24
            plt.plot(time, np.sum(solution[:, self.sim_obj.model.c_idx["d"] *
                                              self.sim_obj.n_ag:self.sim_obj.model.c_idx["d"] *
                                                                    self.sim_obj.n_ag + 5], axis=1),
                     label="0-24", color="black", linewidth=3.0)
            plt.plot(time, np.sum(solution_cmx[:, self.sim_obj.model.c_idx["d"] *
                                                  self.sim_obj.n_ag:self.sim_obj.model.c_idx["d"] *
                                                                self.sim_obj.n_ag + 5], axis=1),
                     label="0-24", color="black", linewidth=3.0, linestyle='--')

            plt.legend()
            plt.gca().set_xlabel('DAYS')
            plt.gca().set_ylabel('deaths')
            plt.gcf().set_size_inches(12, 8)
            plt.tight_layout()
            plt.title(plot_title, y=1.03, fontsize=15)
            title_list = filename_without_ext.split("_")
            plt.savefig('./sens_data/death/' + filename_to_save + '.pdf', format="pdf", bbox_inches='tight')
            plt.close()

    def plot_death_from_model(self, params, cm, filename_without_ext):
        title_list = filename_without_ext.split("_")
        plot_title = 'Target: R0, Susceptibility=' + title_list[0] + ', R0=' + title_list[1]
        self.plot_model(params=params, cm=cm, filename_without_ext=filename_without_ext, filename_to_save=
                        "death_" + filename_without_ext, plot_title=plot_title)
