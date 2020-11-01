from matplotlib import pyplot as plt, cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os
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


def plot_prcc_values(param_list, prcc_vector, filename_without_ext, filename_to_save):
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
    plt.title('PRCC results for ' + title_list[-1] + ' version with children susceptibility ' + title_list[1] +
              ' and base R0=' + title_list[2], y=1.03, fontsize=25)
    plt.savefig('./sens_data/PRCC_bars/' + filename_to_save + '.pdf', format="pdf", bbox_inches='tight')
    plt.close()


def generate_prcc_plots():
    for root, dirs, files in os.walk("./sens_data/simulations"):
        for filename in files:
            filename_without_ext = os.path.splitext(filename)[0]
            print(filename_without_ext)
            saved_simulation = np.loadtxt("./sens_data/simulations/" + filename, delimiter=';')
            saved_simulation = saved_simulation[:, 0:-16]
            upper_tri_indexes = np.triu_indices(16)
            # PRCC analysis for R0
            names = generate_axis_label()
            prcc_list = prcc.get_prcc_values(np.delete(saved_simulation, -2, axis=1))
            plot_prcc_values(np.array(names[upper_tri_indexes]).flatten().tolist(), prcc_list,
                             filename_without_ext, "PRCC_bars_" + filename_without_ext + "_R0")
            plot_prcc_values_as_heatmap(prcc_list, filename_without_ext,
                                        "PRCC_matrix_" + filename_without_ext + "_R0")
            # PRCC analysis for ICU maximum
            prcc_list = prcc.get_prcc_values(np.delete(saved_simulation, -1, axis=1))
            plot_prcc_values(np.array(names[upper_tri_indexes]).flatten().tolist(), prcc_list,
                             filename_without_ext, "PRCC_bars_" + filename_without_ext + "_ICU")
            plot_prcc_values_as_heatmap(prcc_list, filename_without_ext,
                                        "PRCC_matrix_" + filename_without_ext + "_ICU")
