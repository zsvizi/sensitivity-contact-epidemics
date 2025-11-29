import numpy as np


def get_rectangular_matrix_from_upper_triu(rvector: np.ndarray, matrix_size: int) -> np.ndarray:
    """
    Reconstructs a symmetric square matrix from a flattened upper-triangular vector.

    Given a 1D vector representing the upper-triangular elements of a square matrix, this function fills
    a square matrix of size `matrix_size` with these values, ensuring symmetry by mirroring across the diagonal.

    :param np.ndarray rvector: 1D array containing upper-triangular elements of the matrix.
    :param int matrix_size: Size (number of rows/columns) of the resulting square matrix.
    :return np.ndarray: Symmetric square matrix of shape (matrix_size, matrix_size).
    """
    # Get indices of the upper-triangular part of the matrix (including diagonal)
    upper_tri_indexes = np.triu_indices(matrix_size)

    # Initialize a zero matrix of desired size
    new_contact_mtx = np.zeros((matrix_size, matrix_size))

    # Fill the upper-triangular part with the values from rvector
    new_contact_mtx[upper_tri_indexes] = rvector

    # Mirror the upper-triangular part to the lower-triangular part to ensure symmetry
    new_2 = new_contact_mtx.T
    new_2[upper_tri_indexes] = rvector

    # Return the reconstructed symmetric matrix
    return np.array(new_2)


def get_prcc_values(lhs_output_table: np.ndarray) -> np.ndarray:
    """
    Computes Partial Rank Correlation Coefficients (PRCC) for a given LHS output table.

    This function:
      1. Ranks each column of the LHS output table.
      2. Computes the correlation matrix of the ranked variables.
      3. Inverts the correlation matrix
      4. Computes the PRCC for each input parameter relative to the output variable
         (assumed to be the last column in the table).

    :param np.ndarray lhs_output_table: LHS sample table with parameters in columns and
                                        simulation output in the last column.
    :return np.ndarray: 1D array of PRCC values, one for each input parameter.
    """
    # Rank the columns of the LHS output table
    ranked = (lhs_output_table.argsort(0)).argsort(0)

    # Compute correlation matrix of ranked values
    corr_mtx = np.corrcoef(ranked.T)
    # The transpose ensures that rows correspond to variables and columns to observations

    # Invert the correlation matrix (or use pseudo-inverse if singular)
    if np.linalg.det(corr_mtx) < 1e-50:  # Check if correlation matrix is nearly singular
        corr_mtx_inverse = np.linalg.pinv(corr_mtx)  # Pseudo-inverse is more stable
    else:
        corr_mtx_inverse = np.linalg.inv(corr_mtx)

    # Compute PRCC values for each input parameter relative to the output
    parameter_count = lhs_output_table.shape[1] - 1  # Exclude the output column
    prcc_vector = np.zeros(parameter_count)

    for w in range(parameter_count):
        # PRCC formula: correlation between parameter w and output, controlling for other parameters
        prcc_vector[w] = -corr_mtx_inverse[w, parameter_count] / \
                         np.sqrt(corr_mtx_inverse[w, w] *
                                 corr_mtx_inverse[parameter_count, parameter_count])

    return prcc_vector
