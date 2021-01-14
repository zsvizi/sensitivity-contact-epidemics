import os

from matplotlib import pyplot as plt, cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from dataloader import DataLoader
import prcc

plt.style.use('seaborn-whitegrid')


def generate_axis_label():
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


def plot_prcc_values_as_heatmap(prcc_vector, filename_without_ext, filename):
    """
    Plots a given PRCC result as a heatmap
    :param filename: name of the saved file
    :param prcc_vector: list of PRCC values
    :return: None
    """
    os.makedirs("./sens_data/PRCC_bars", exist_ok=True)
    number_of_age_groups = 16
    upper_tri_indexes = np.triu_indices(number_of_age_groups)
    new_contact_mtx = np.zeros((number_of_age_groups, number_of_age_groups))
    new_contact_mtx[upper_tri_indexes] = prcc_vector
    new_2 = new_contact_mtx.T
    new_2[upper_tri_indexes] = prcc_vector
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    plt.margins(0, tight=False)
    cmap = mcolors.LinearSegmentedColormap.from_list("", [mcolors.CSS4_COLORS["lightgrey"],
                                                          mcolors.CSS4_COLORS["crimson"]])
    param_list = range(0, number_of_age_groups, 1)
    corr = pd.DataFrame(new_2, columns=param_list, index=param_list)
    plt.figure(figsize=(12, 12))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    title_list = filename_without_ext.split("_")
    plt.title('PRCC results for ' + title_list[-1] + ' version with\nchildren susceptibility ' + title_list[1] +
              ' and base R0=' + title_list[2], y=1.03, fontsize=25)
    plt.imshow(corr, cmap=cmap, origin='lower')
    plt.colorbar(pad=0.02, fraction=0.04)
    plt.grid(b=None)
    plt.gca().set_xticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
    plt.gca().set_yticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
    plt.gca().grid(which='minor', color='w', linestyle='-', linewidth=2)
    plt.xticks(ticks=param_list, labels=param_list)
    plt.yticks(ticks=param_list, labels=param_list)
    plt.savefig('./sens_data/PRCC_bars/' + filename + '.pdf', cmap=cmap, format="pdf",
                bbox_inches='tight')
    plt.close()


def plot_prcc_values_lockdown_3(prcc_vector, filename_to_save, plot_title):
    """
    Plots a given PRCC result
    :param filename_to_save: name of the output file
    :param prcc_vector: list of PRCC values
    :return: None
    """
    os.makedirs("./sens_data/PRCC_bars", exist_ok=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    plt.margins(0, tight=False)
    parameter_count = len(list(prcc_vector)) // 3

    age_group_vector = list(range(parameter_count))*3
    contact_type_vector = ["school"]*16 + ["work"]*16 + ["other"]*16

    data_to_df = [[age_group_vector[i], contact_type_vector[i], prcc_vector[i]] for i in range(parameter_count * 3)]

    df = pd.DataFrame(data_to_df, columns=['Age group', 'Contact type', 'val'])

    plt.figure(figsize=(17, 10))

    colors = {"home": "#ff96da", "work": "#96daff", "school": "#96ffbb", "other": "#ffbb96", "total": "blue"}

    dftemp = df.pivot("Age group", "Contact type", "val")
    dftemp = dftemp[["school", "work", "other"]]
    dftemp.plot(kind='bar', width=0.85, color=[colors["school"], colors["work"], colors["other"]])

    xp = range(parameter_count)

    param_list = map(lambda x: "75+" if x == 15 else str(5*x) + "-"+str(5*x+4), range(parameter_count))

    plt.tick_params(direction="in")
    plt.xticks(ticks=xp, labels=param_list, rotation=45)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.ylabel('PRCC indices', labelpad=10, fontsize=20)
    plt.xlabel('Age groups', labelpad=10, fontsize=20)
    plt.title(plot_title, y=1.03, fontsize=25)
    plt.savefig('./sens_data/PRCC_bars/' + filename_to_save + '.pdf', format="pdf", bbox_inches='tight')
    plt.close()


def plot_prcc_values(param_list, prcc_vector, filename_without_ext, filename_to_save, plot_title):
    """
    Plots a given PRCC result
    :param filename_to_save: name of the output file
    :param filename_without_ext: name to identify the saved file
    :param param_list: list of parameter names
    :param prcc_vector: list of PRCC values
    :return: None
    """
    os.makedirs("./sens_data/PRCC_bars", exist_ok=True)
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
    plt.savefig('./sens_data/PRCC_bars/' + filename_to_save + '.pdf', format="pdf", bbox_inches='tight')
    plt.close()


def generate_prcc_plots(sim_obj):
    n_ag = sim_obj.no_ag
    upp_tri_size = int((n_ag + 1) * n_ag / 2)
    sim_folder = "simulations_after_redei"
    lhs_folder = "lhs"
    cm_total_full = sim_obj.contact_matrix * sim_obj.age_vector
    for root, dirs, files in os.walk("./sens_data/" + sim_folder):
        for filename in files:
            filename_without_ext = os.path.splitext(filename)[0]
            print(filename_without_ext)
            saved_simulation = np.loadtxt("./sens_data/" + sim_folder + "/" + filename,
                                          delimiter=';')
            saved_lhs_values = np.loadtxt("./sens_data/" + lhs_folder + "/" + filename.replace("simulations", "lhs"),
                                          delimiter=';')
            if 'lockdown_3' in filename_without_ext:
                sim_data = saved_lhs_values[:, :3 * upp_tri_size]
                # Transform sim_data to get positively correlating variables
                # Here ratio to subtract is negatively correlated to the targets, thus
                # 1 - ratio (i.e. ratio of remaining contacts) is positively correlated
                sim_data = 1 - sim_data
                names = [str(i) for i in range(3 * n_ag)]
            elif "lockdown" in filename_without_ext:
                sim_data = saved_lhs_values[:, :(n_ag*(n_ag+1)) // 2]
                sim_data = 1 - sim_data
                upper_tri_indexes = np.triu_indices(n_ag)
                names = generate_axis_label()[upper_tri_indexes]
            elif "mitigation" in filename_without_ext:
                sim_data = saved_lhs_values[:, :(n_ag * (n_ag + 1)) // 2]
                upper_tri_indexes = np.triu_indices(n_ag)
                names = generate_axis_label()[upper_tri_indexes]
            else:
                raise Exception('Matrix type is unknown!')

            # PRCC analysis for R0
            simulation = np.append(sim_data, saved_simulation[:, -n_ag - 1].reshape((-1, 1)), axis=1)
            prcc_list = prcc.get_prcc_values(simulation)
            if 'lockdown_3' in filename_without_ext:
                prcc_matrix_school = prcc.get_rectangular_matrix_from_upper_triu(prcc_list[:upp_tri_size], n_ag)
                prcc_matrix_work = prcc.get_rectangular_matrix_from_upper_triu(prcc_list[upp_tri_size:2*upp_tri_size],
                                                                               n_ag)
                prcc_matrix_other = prcc.get_rectangular_matrix_from_upper_triu(prcc_list[2*upp_tri_size:], n_ag)

                agg_prcc_school = \
                    (prcc_matrix_school.dot(sim_obj.age_vector) / np.sum(sim_obj.age_vector)).flatten()
                agg_prcc_work = \
                    (prcc_matrix_work.dot(sim_obj.age_vector) / np.sum(sim_obj.age_vector)).flatten()
                agg_prcc_other = \
                    (prcc_matrix_other.dot(sim_obj.age_vector) / np.sum(sim_obj.age_vector)).flatten()
                prcc_list = np.array([agg_prcc_school, agg_prcc_work, agg_prcc_other]).flatten()

                title_list = filename_without_ext.split("_")
                plot_title = 'Target: R0, Susceptibility=' + title_list[2] + ', R0=' + title_list[3]
                plot_prcc_values_lockdown_3(prcc_list, "PRCC_bars_" + filename_without_ext + "_R0", plot_title)
            elif 'lockdown' in filename_without_ext or 'mitigation' in filename_without_ext:
                title_list = filename_without_ext.split("_")
                plot_title = 'Target: R0, Susceptibility=' + title_list[2] + ', R0=' + title_list[3]
                plot_prcc_values(np.array(names).flatten().tolist(), prcc_list,
                                 filename_without_ext, "PRCC_bars_" + filename_without_ext + "_R0", plot_title)

            # PRCC analysis for ICU maximum
            simulation = np.append(sim_data, saved_simulation[:, -n_ag - 2].reshape((-1, 1)), axis=1)
            prcc_list = prcc.get_prcc_values(simulation)
            if 'lockdown_3' in filename_without_ext:
                prcc_matrix_school = prcc.get_rectangular_matrix_from_upper_triu(prcc_list[:upp_tri_size], n_ag)
                prcc_matrix_work = prcc.get_rectangular_matrix_from_upper_triu(prcc_list[upp_tri_size:2 * upp_tri_size],
                                                                               n_ag)
                prcc_matrix_other = prcc.get_rectangular_matrix_from_upper_triu(prcc_list[2 * upp_tri_size:], n_ag)

                agg_prcc_school = \
                    (prcc_matrix_school.dot(sim_obj.age_vector) / np.sum(sim_obj.age_vector)).flatten()
                agg_prcc_work = \
                    (prcc_matrix_work.dot(sim_obj.age_vector) / np.sum(sim_obj.age_vector)).flatten()
                agg_prcc_other = \
                    (prcc_matrix_other.dot(sim_obj.age_vector) / np.sum(sim_obj.age_vector)).flatten()
                prcc_list = np.array([agg_prcc_school, agg_prcc_work, agg_prcc_other]).flatten()

                title_list = filename_without_ext.split("_")
                plot_title = 'Target: ICU, Susceptibility=' + title_list[2] + ', R0=' + title_list[3]
                plot_prcc_values_lockdown_3(prcc_list, "PRCC_bars_" + filename_without_ext + "_ICU", plot_title)
            elif 'lockdown' in filename_without_ext or 'mitigation' in filename_without_ext:
                title_list = filename_without_ext.split("_")
                plot_title = 'Target: ICU, Susceptibility=' + title_list[2] + ', R0=' + title_list[3]
                plot_prcc_values(np.array(names).flatten().tolist(), prcc_list,
                                 filename_without_ext, "PRCC_bars_" + filename_without_ext + "_ICU", plot_title)


def plot_symm_contact_matrix_as_bars(param_list, contact_vector, file_name):
    """
    Plots a given contact matrix like as barplot
    :param param_list: list of names for the corresponding contacts
    :param contact_vector: list of contact numbers
    :param file_name: filename to save the figure
    :return: None
    """
    ymax = max(contact_vector)
    color_map = cm.get_cmap('Blues')
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


def plot_2d_contact_matrices():
    """
    Plots the contact matrices of the given country as a heatmap with different colors
    """
    data = DataLoader()
    age_distribution = data.age_data.reshape((-1, 1))
    colors = {"home": "#ff96da", "work": "#96daff", "school": "#96ffbb", "other": "#ffbb96", "total": "blue"}
    contact_total = np.array([data.contact_data[t] for t in list(colors.keys())[:-1]]).sum(axis=0)
    for t in colors.keys():
        contact_small = data.contact_data[t] if t != "total" else contact_total
        param_list = range(0, 16, 1)
        corr = pd.DataFrame(contact_small * age_distribution, columns=param_list, index=param_list)
        plt.figure(figsize=(12, 12))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.title('Symmetric ' + t + ' contact matrix of Hungary', y=1.03, fontsize=25)
        plt.imshow(corr, cmap=mcolors.LinearSegmentedColormap.from_list("", ["white",
                                                                             colors[t]]),
                   origin='lower')
        plt.colorbar(pad=0.02, fraction=0.04)
        plt.grid(b=None)
        number_of_age_groups = 16
        plt.gca().set_xticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
        plt.gca().set_yticks(np.arange(0.5, number_of_age_groups, 1), minor=True)
        plt.gca().grid(which='minor', color='gray', linestyle='-', linewidth=1)
        plt.xticks(ticks=param_list, labels=param_list)
        plt.yticks(ticks=param_list, labels=param_list)
        plt.savefig('./sens_data/CM_symm_Hungary' + "_" + t + '.pdf', format="pdf",
                    bbox_inches='tight')
        plt.close()


def generate_stacked_plots():
    names = generate_axis_label()
    upper_tri_indexes = np.triu_indices(16)
    data = DataLoader()
    contact_matrix = data.contact_data["work"] + data.contact_data["home"] + \
                     data.contact_data["other"] + data.contact_data["school"]
    age_vector = data.age_data.reshape((-1, 1))
    symm_cm_total = contact_matrix * age_vector
    plot_symm_contact_matrix_as_bars(np.array(names[upper_tri_indexes]).flatten().tolist(),
                                     symm_cm_total[upper_tri_indexes],
                                     file_name="CM_symm_Hungary")

    # Plot stacked home-work-school-total
    c_matrices = []
    c_names = np.array(["home", "school", "work", "other"])
    color_list = ["#ff96da", "#96ffbb", "#96daff", "#ffbb96"]
    for n in c_names:
        contact_matrix = data.contact_data[n]
        age_distribution = data.age_data.reshape((-1, 1))
        symm_cm = contact_matrix * age_distribution
        c_matrices.append(symm_cm[upper_tri_indexes])
    plot_stacked(c_matrices, np.array(names[upper_tri_indexes]).flatten().tolist(), c_names,
                 color_list=color_list, title="Stacked version of different types of contacts",
                 filename="HWSO_contacts")

    # Plot stacked total-normed-home
    c_matrices = []
    c_names = ["home", "normed", "total"]
    color_list = ["#ff96da", "#ecff7f", "#56a8ff"]
    symm_cm_1 = data.contact_data["home"] * age_vector
    symm_cm_2 = symm_cm_total - np.min(symm_cm_total - symm_cm_1)
    c_matrices.append(symm_cm_1[upper_tri_indexes])
    c_matrices.append(symm_cm_2[upper_tri_indexes] - symm_cm_1[upper_tri_indexes])
    c_matrices.append(symm_cm_total[upper_tri_indexes] - symm_cm_2[upper_tri_indexes])
    plot_stacked(c_matrices, np.array(names[upper_tri_indexes]).flatten().tolist(), c_names,
                 color_list=color_list, title="Compound version of total, normed and home contacts",
                 filename="TNH_contacts")


def plot_solution_ic(obj, time, params, cm_list, legend_list, title_part):
    os.makedirs("./sens_data/dinamics", exist_ok=True)
    # for comp in compartments:
    for idx, cm in enumerate(cm_list):
        solution = obj.model.get_solution(t=time, parameters=params, cm=cm)
        plt.plot(time, np.sum(solution[:,
                              obj.model.c_idx["ic"] * obj.no_ag:(obj.model.c_idx["ic"] + 1) * obj.no_ag],
                              axis=1),
                 label=legend_list[idx])
    plt.legend()
    plt.gca().set_xlabel('days')
    plt.gca().set_ylabel('ICU usage')
    plt.gcf().set_size_inches(6, 6)
    plt.tight_layout()
    plt.savefig('./sens_data/dinamics/solution_ic' + "_" + title_part + '.pdf', format="pdf")
    plt.close()


def plot_solution_inc(obj, time, params, cm_list, legend_list, title_part):
    os.makedirs("./sens_data/dinamics", exist_ok=True)
    # for comp in compartments:
    for legend, cm in zip(legend_list, cm_list):
        solution = obj.model.get_solution(t=time, parameters=params, cm=cm)
        plt.plot(time[:-1],
                 np.diff(np.sum(solution[:,
                                obj.model.c_idx["c"] * obj.no_ag:(obj.model.c_idx["c"] + 1) * obj.no_ag],
                                axis=1)),
                 label=legend)
    plt.legend()
    plt.gca().set_xlabel('days')
    plt.gca().set_ylabel('Incidence')
    plt.gcf().set_size_inches(6, 6)
    plt.tight_layout()
    plt.savefig('./sens_data/dinamics/solution_inc' + "_" + title_part + '.pdf', format="pdf")
    plt.close()


def plot_different_susc(file_list, c_list, l_list, title):
    os.makedirs("./sens_data/dinamics", exist_ok=True)
    for idx, f in enumerate(file_list):
        sol = np.loadtxt('./sens_data/dinamics/' + f)
        plt.plot(np.arange(len(sol)) / 2, sol, c_list[idx], label=l_list[idx])
    plt.legend(loc='upper left')
    plt.gca().set_xlabel('days')
    plt.gca().set_ylabel('ICU usage')
    plt.tight_layout()
    plt.savefig('./sens_data/dinamics/' + title + '.pdf', format="pdf")
    plt.close()


def plot_contact_matrix_as_grouped_bars():
    os.makedirs("./sens_data", exist_ok=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    plt.margins(0, tight=False)

    parameter_count = 16
    xp = range(parameter_count)
    param_list = list(map(lambda x: "75+" if x == 15 else str(5 * x) + "-" + str(5 * x + 4), range(parameter_count)))

    age_group_vector = list(range(parameter_count))*3
    contact_type_vector = ["school"]*16 + ["work"]*16 + ["other"]*16

    data = DataLoader()
    age_distribution = data.age_data.reshape((-1, 1))

    colors = {"home": "#ff96da", "work": "#96daff", "school": "#96ffbb", "other": "#ffbb96", "total": "blue"}

    contact_school = np.sum(data.contact_data["school"] * age_distribution, axis=1)
    contact_work = np.sum(data.contact_data["work"] * age_distribution, axis=1)
    contact_other = np.sum(data.contact_data["other"] * age_distribution, axis=1)
    contacts_as_vector = np.array([contact_school, contact_work, contact_other]).flatten()

    plt.figure(figsize=(17, 10))
    data_to_df = [[age_group_vector[i], contact_type_vector[i], contacts_as_vector[i]] for i in range(parameter_count * 3)]
    df = pd.DataFrame(data_to_df, columns=['Age group', 'Contact type', 'val'])
    dftemp = df.pivot("Age group", "Contact type", "val")
    dftemp = dftemp[["school", "work", "other"]]
    dftemp.plot(kind='bar', width=0.85, color=[colors["school"], colors["work"], colors["other"]])

    plt.tick_params(direction="in")
    plt.xticks(ticks=xp, labels=param_list, rotation=45)
    plt.xlabel('Age groups', labelpad=10, fontsize=20)
    plt.title("Contacts from symmetric CMs", y=1.03, fontsize=25)
    plt.savefig('./sens_data/CM_symm_groupped.pdf', format="pdf", bbox_inches='tight')
    plt.close()

    contact_school = np.sum(data.contact_data["school"], axis=1)
    contact_work = np.sum(data.contact_data["work"], axis=1)
    contact_other = np.sum(data.contact_data["other"], axis=1)
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
    plt.title("Contacts from original CMs", y=1.03, fontsize=25)
    plt.savefig('./sens_data/CM_prem_groupped.pdf', format="pdf", bbox_inches='tight')
    plt.close()