import numpy as np
from scipy.linalg import block_diag

from src.model.r0_generator_base import R0GeneratorBase


class R0ValidationModel(R0GeneratorBase):
    """
    A reproduction number (R_0) generator for validation purposes.

    This class implements a basic SEIR-like structure for testing the R_0 computation
    framework. It uses the next-generation matrix (NGM) approach.
    """

    def __init__(self, param: dict, country: str = "validation", n_age: int = 3) -> None:
        """
        Initializes the validation R_0 model with epidemiological parameters.

        :param dict param: Dictionary containing model parameters such as:
            - alpha : rate of progression from exposed -> infectious
            - gamma : recovery rate
            - susc  : age-specific susceptibility vector
        :param str country: Country or model label
        :param int n_age: Number of age groups
        """
        self.country = country
        state = ["e", "i"]  # Model includes 'exposed' and 'infectious' compartments

        super().__init__(param=param, states=state, n_age=n_age)

        # Initialize matrices used in next-generation computation
        self._get_e()
        self._get_v()

    def _get_v(self) -> np.ndarray:
        """
        Constructs and invert the V matrix (transition matrix).

        The V matrix captures movement between infectious compartments:
            - E -> E : removal from exposure at rate α
            - E -> I : progression from exposed to infectious (−α)
            - I -> I : removal from infectious at rate γ

        :return np.ndarray: The inverse of the constructed V matrix
        """
        idx = self._idx
        v = np.zeros((self.n_age * self.n_states, self.n_age * self.n_states))

        # E -> E (removal)
        v[idx("e"), idx("e")] = self.parameters["alpha"]
        # E -> I (progression)
        v[idx("i"), idx("e")] = -self.parameters["alpha"]
        # I -> I (removal)
        v[idx("i"), idx("i")] = self.parameters["gamma"]

        # Store inverse for use in R_0 computation
        self.v_inv = np.linalg.inv(v)
        return self.v_inv

    def _get_f(self, cm: np.ndarray) -> np.ndarray:
        """
        Constructs the F matrix (new infection matrix).

        The F matrix represents the rate of new exposures caused by contact
        between infectious and susceptible individuals, weighted by the contact matrix.

        :param np.ndarray cm: Contact matrix representing average contact between age groups

        :return np.ndarray: The constructed F matrix of size (n_age * n_states, n_age * n_states).
        """
        i = self.i  # index dictionary for states
        s_mtx = self.s_mtx  # scaling for matrix positioning
        n_state = self.n_states
        susc_vec = self.parameters["susc"].reshape((-1, 1))  # susceptibility vector by age

        f = np.zeros((self.n_age * n_state, self.n_age * n_state))

        # F(e,i): new exposures caused by infecteds in each age group
        f[i["e"]:s_mtx:n_state, i["i"]:s_mtx:n_state] = cm.T * susc_vec

        return f

    def _get_e(self):
        """
        Construct the E matrix block used to structure the next-generation matrix.

        This method creates a block-diagonal matrix indicating which compartments
        correspond to the "exposed" (E) state across all age groups.
        """
        block = np.zeros(self.n_states)
        block[0] = 1  # mark 'E' state
        self.e = block

        # Build a block-diagonal structure for multiple age groups
        for _ in range(1, self.n_age):
            self.e = block_diag(self.e, block)
