import numpy as np
from scipy.linalg import block_diag

from src.model.r0_generator_base import R0GeneratorBase


class R0SeirSVModel(R0GeneratorBase):
    """
    R_0 generator for the SEIR model variant with seasonality and vaccination structure (UK model).

    This class computes the next generation matrix (NGM) and its dominant eigenvalue, which
    represents the basic reproduction number (R_0).
    """

    def __init__(self, param: dict, country: str = "UK", n_age: int = 15) -> None:
        """
        Initializes the SEIR-based R_0 generator for the UK model.

        :param dict param: Dictionary of model parameters including 'gamma', 'rho', and 'susc'.
        :param str country: Country code used for model identification (default: "UK").
        :param int n_age: Number of age groups used to construct the contact matrix (default: 15).
        """
        self.country = country

        # Define the model states
        state = ["e", "i"]
        super().__init__(param=param, states=state, n_age=n_age)

        # Initialize transformation matrices E and V
        self._get_e()
        self._get_v()

    def _get_v(self) -> np.ndarray:
        """
        Construct the V matrix (transitions between infected compartments), compute its inverse
        and stored as self.v_inv (np.ndarray)

        The matrix V describes rates of progression and recovery within infected
        compartments. The structure is based on the following transitions:

        - E -> E: loss due to progression (γ)
        - E -> I: gain in I due to γ
        - I -> I: loss due to recovery (ρ)
        """
        idx = self._idx
        v = np.zeros((self.n_age * self.n_states, self.n_age * self.n_states))

        # E compartment: rate of leaving E (progression to I)
        v[idx("e"), idx("e")] = self.parameters["gamma"]

        # Transition from E to I
        v[idx("i"), idx("e")] = -self.parameters["gamma"]

        # I compartment: rate of recovery
        v[idx("i"), idx("i")] = self.parameters["rho"]

        # Compute inverse V⁻¹ for next generation matrix calculation
        self.v_inv = np.linalg.inv(v)

    def _get_f(self, cm: np.ndarray) -> np.ndarray:
        """
        Constructs the F matrix representing new infections in each compartment.

        The F matrix captures the rate of appearance of new infections caused
        by infectious individuals interacting via the contact matrix.

        :param np.ndarray cm: Contact matrix representing mixing patterns between age groups.
        :return np.ndarray: F matrix used in next generation matrix computation.
        """
        i = self.i
        s_mtx = self.s_mtx
        n_state = self.n_states

        # Age-dependent susceptibility vector (susc)
        susc_vec = self.parameters["susc"].reshape((-1, 1))

        # Initialize F matrix
        f = np.zeros((self.n_age * n_state, self.n_age * n_state))

        # Infection terms: both E and I contribute to transmission in this model variant
        f[i["e"]:s_mtx:n_state, i["e"]:s_mtx:n_state] = cm.T * susc_vec
        f[i["i"]:s_mtx:n_state, i["i"]:s_mtx:n_state] = cm.T * susc_vec

        return f

    def _get_e(self) -> None:
        """
        Construct the E selection matrix used for age-structured transformation.

        This matrix defines how the next generation matrix is projected between
        compartments and age groups. It is built as a block diagonal structure
        with ones in the "E" position for each age group.
        """
        # Base block with a 1 for the E compartment
        block = np.zeros(self.n_states)
        block[0] = 1
        self.e = block

        # Build block-diagonal structure for all age groups
        for _ in range(1, self.n_age):
            self.e = block_diag(self.e, block)
