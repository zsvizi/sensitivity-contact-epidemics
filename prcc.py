import numpy as np
from smt.sampling_methods import LHS


def get_contact_matrix_from_upper_triu(rvector, age_vector):
    upper_tri_indexes = np.triu_indices(age_vector.shape[0])
    new_contact_mtx = np.zeros((age_vector.shape[0], age_vector.shape[0]))
    new_contact_mtx[upper_tri_indexes] = rvector
    new_2 = new_contact_mtx.T
    new_2[upper_tri_indexes] = rvector
    vector = np.array(new_2 / age_vector)
    return vector


def get_prcc_input(lhs_vector: np.ndarray, cm: np.ndarray):
    cm_total = cm - lhs_vector.reshape((-1, 1))
    return np.sum(cm_total, axis=1)


def create_latin_table(n_of_samples, lower, upper):
    bounds = np.array([lower, upper]).T
    sampling = LHS(xlimits=bounds)
    return sampling(n_of_samples)


def get_prcc_values(lhs_output_table):
    """
    Creates the PRCC values of last column of an ndarray depending on the columns before.
    :param lhs_output_table: ndarray
    :return: ndarray
    """
    ranked = (lhs_output_table.argsort(0)).argsort(0)
    corr_mtx = np.corrcoef(ranked.T)
    if np.linalg.det(corr_mtx) < 1e-50:  # determine if singular
        corr_mtx_inverse = np.linalg.pinv(corr_mtx)  # may need to use pseudo inverse
    else:
        corr_mtx_inverse = np.linalg.inv(corr_mtx)

    parameter_count = lhs_output_table.shape[1] - 1

    prcc_vector = np.zeros(parameter_count)
    for w in range(parameter_count):  # compute PRCC btwn each param & sim result
        prcc_vector[w] = -corr_mtx_inverse[w, parameter_count] / \
                         np.sqrt(corr_mtx_inverse[w, w] *
                                 corr_mtx_inverse[parameter_count, parameter_count])
    return prcc_vector
