import os
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataloader import DataLoader
from src.simulation_npi import SimulationNPI

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
        plt.bar(xp, list(prcc_vector), align='center', color=cmap(prcc_vector))

        plt.xticks(ticks=xp, labels=param_list, rotation=90)
        plt.yticks(ticks=np.arange(-1, 1.05, 0.1))
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.ylabel('PRCC indices', labelpad=10, fontsize=20)
        plt.xlabel('Pairs of age groups', labelpad=10, fontsize=20)
        title_list = filename_without_ext.split("_")
        plt.title(plot_title, y=1.03, fontsize=25)
        plt.savefig('./sens_data/PRCC_plot/' + filename_to_save + '.pdf', format="pdf", bbox_inches='tight')
        plt.close()

    def generate_prcc_plots(self, prcc_vector, filename_without_ext):
        title_list = filename_without_ext.split("_")
        plot_title = 'Target: R0, Susceptibility=' + title_list[0] + ', R0=' + title_list[1]
        names = np.array(self.generate_axis_label()[self.sim_obj.upper_tri_indexes])
        self.plot_prcc_values_lockdown_3(prcc_vector, "PRCC_bars_" + filename_without_ext + "_R0", plot_title)
        if "lockdown_3" not in filename_without_ext:
            self.plot_prcc_values(names.flatten().tolist(), prcc_vector, filename_without_ext,
                                  "PRCC_plot_" + filename_without_ext + "_R0", plot_title)

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
        for t in range(1, len(list_of_flattened_matrices)):
            plt.bar(xp, list(list_of_flattened_matrices[t]), align='center',
                    bottom=np.array(list_of_flattened_matrices[:t]).sum(axis=0), label=labels[t], color=color_list[t])
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
