from abc import ABC, abstractmethod

import numpy as np


class R0GeneratorBase(ABC):
    """
    Abstract base class for generating the basic reproduction number (R_0)
    or the effective reproduction number based on the next-generation matrix (NGM)
    in age-structured compartmental epidemic models.
    """

    def __init__(self, param: dict, states: list, n_age: int) -> None:
        """
        Initialize the R_0 generator base class.

        :param dict param: Dictionary containing model parameters (e.g., transition rates, probabilities)
        :param list states: List of disease compartments (e.g., ["S", "E", "I", "R"])
        :param int n_age: Number of age groups in the model
        """
        self.states = states
        self.n_age = n_age
        self.parameters = param
        self.n_states = len(self.states)

        # Mapping from compartment/state name to its index
        self.i = {self.states[index]: index for index in np.arange(0, self.n_states)}

        # Total number of state-age combinations
        self.s_mtx = self.n_age * self.n_states

        # Placeholders for model matrices and intermediate results
        self.v_inv = None
        self.e = None
        self.contact_matrix = np.zeros((n_age, n_age))

    def _idx(self, state: str) -> np.ndarray:
        """
        Create a boolean mask for the indices corresponding to the given disease state.

        :param str state: Name of the disease compartment (e.g., "I" for infected).
        :return np.ndarray: Boolean mask selecting positions of the specified state across all age groups.
        """
        return np.arange(self.n_age * self.n_states) % self.n_states == self.i[state]

    def get_eig_val(self, susceptibles: np.ndarray, population: np.ndarray, contact_mtx: np.ndarray = None) -> list:
        """
        Compute the dominant eigenvalue (largest absolute value) of the next-generation matrix (NGM),
        representing the effective reproduction number (R_0).

        :param np.ndarray susceptibles: array of susceptible individuals per age group
        :param np.ndarray population: array of total population per age group
        :param np.ndarray contact_mtx: Contact matrix between age groups.
                                       If None, the internally stored contact matrix is used.
        :return list: List of effective reproduction numbers (R_0)
        """
        # For effective reproduction number: [c_{j,i} * S_i(t) / N_i(t)]
        if contact_mtx is not None:
            self.contact_matrix = contact_mtx

        # Normalize contact matrix by population size
        contact_matrix = self.contact_matrix / population.reshape((-1, 1))

        cm_tensor = np.tile(contact_matrix, (susceptibles.shape[0], 1, 1))
        susc_tensor = susceptibles.reshape((susceptibles.shape[0], susceptibles.shape[1], 1))

        contact_matrix_tensor = cm_tensor * susc_tensor
        eig_val_eff = []

        for cm in contact_matrix_tensor:
            # Build next-generation matrix (NGM)
            f = self._get_f(cm)
            ngm_large = f @ self.v_inv
            ngm = self.e @ ngm_large @ self.e.T

            # Compute eigenvalues and take the dominant one (largest absolute value)
            eig_val = np.sort(list(map(lambda x: np.abs(x), np.linalg.eig(ngm)[0])))
            eig_val_eff.append(float(eig_val[-1]))

        return eig_val_eff

    @abstractmethod
    def _get_e(self) -> np.ndarray:
        """
        Construct the transformation matrix E used in next-generation matrix computation.

        :return np.ndarray: Transformation matrix E.
        """
        pass

    @abstractmethod
    def _get_v(self):
        """
        Construct the inverse of matrix V, representing transitions between infected compartments.

        :return np.ndarray: Inverse of matrix V.
        """
        pass

    @abstractmethod
    def _get_f(self, contact_matrix: np.ndarray) -> np.ndarray:
        """
        Construct the infection (F) matrix representing the rate of new infections
        per compartment and age group.

        :param np.ndarray contact_matrix: Contact matrix between age groups.
        :return np.ndarray: Infection matrix F.
        """
        pass
