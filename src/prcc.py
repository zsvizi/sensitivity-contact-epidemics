import numpy as np
from numpy import ndarray
from scipy.stats import beta, t


def get_rectangular_matrix_from_upper_triu(rvector, matrix_size) -> np.ndarray:
    upper_tri_indexes = np.triu_indices(matrix_size)
    new_contact_mtx = np.zeros((matrix_size, matrix_size))
    new_contact_mtx[upper_tri_indexes] = rvector
    new_2 = new_contact_mtx.T
    new_2[upper_tri_indexes] = rvector
    return np.array(new_2)


def get_contact_matrix_from_upper_triu(rvector, age_vector) -> np.ndarray:
    new_2 = get_rectangular_matrix_from_upper_triu(rvector=rvector,
                                                   matrix_size=age_vector.shape[0])
    vector = np.array(new_2 / age_vector)
    return vector


def get_prcc_input(lhs_vector: np.ndarray, cm: np.ndarray):
    cm_total = cm - lhs_vector.reshape((-1, 1))
    return np.sum(cm_total, axis=1)


def get_prcc_values(lhs_output_table: np.ndarray, number_of_samples: int) -> ndarray:
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

        # p-values Size (136) for lockdown and 408 for lockdown3  [formula based on Simeone Marino, Ian B. Hogue paper]
        T = prcc_vector * np.sqrt((number_of_samples - 2 - parameter_count) / (1 - prcc_vector ** 2))
        # p-value for 2-sided test
        dof = number_of_samples - 2 - parameter_count
        p_value = 2 * (1 - t.cdf(abs(T), dof))

    return prcc_vector



