import numpy as np
from scipy.linalg import block_diag

from src.model.r0_generator_base import R0GeneratorBase


class R0Generator(R0GeneratorBase):
    """
    States per age group:
    - l1, l2 : two-stage latent period
    - ip     : presymptomatic infectious
    - a1,a2,a3 : three-stage asymptomatic infectious
    - i1,i2,i3 : three-stage symptomatic infectious
    """

    def __init__(self, param: dict, n_age: int = 16,
                 country: str = "Hungary") -> None:
        """
        :param dict param: Parameter dictionary (progression rates, susceptibilities, etc.)
        :param int n_age: Number of age groups
        :param str country: Country identifier for parameter presets
        """
        self.country = country

        states = ["l1", "l2", "ip", "a1", "a2", "a3", "i1", "i2", "i3"]
        super().__init__(param=param, states=states, n_age=n_age)

        # Number of substages in each progression chain
        self.n_l = 2  # latent stages
        self.n_a = 3  # asymptomatic stages
        self.n_i = 3  # symptomatic stages

        self._get_e()
        self._get_v()

    def _get_v(self) -> np.ndarray:
        """
        Constructs the V matrix (transitions out of infected compartments)
        for the next-generation matrix approach. After construction, V is inverted and stored as self.v_inv.

        V matrix:
        - diagonal: rate of leaving each compartment
        - off-diagonal: negative transition rates into next-stage compartments
        """
        idx = self._idx  # helper: get flattened index(position) for "state"
        v = np.zeros((self.n_age * self.n_states, self.n_age * self.n_states))

        # LATENT STAGES: L1 -> L2
        v[idx("l1"), idx("l1")] = self.n_l * self.parameters["alpha_l"]
        v[idx("l2"), idx("l1")] = -self.n_l * self.parameters["alpha_l"]

        # L2 -> Ip
        v[idx("l2"), idx("l2")] = self.n_l * self.parameters["alpha_l"]
        v[idx("ip"), idx("l2")] = -self.n_l * self.parameters["alpha_l"]

        # Presymptomatic -> I1/A1 split
        v[idx("ip"), idx("ip")] = self.parameters["alpha_p"]
        v[idx("i1"), idx("ip")] = -self.parameters["alpha_p"] * (1 - self.parameters["p"])  # Symptomatic
        v[idx("a1"), idx("ip")] = -self.parameters["alpha_p"] * self.parameters["p"]  # Asymptomatic

        # Asymptomatic: A1 -> A2 -> A3 to recovery
        v[idx("a1"), idx("a1")] = self.n_a * self.parameters["gamma_a"]
        v[idx("a2"), idx("a1")] = -self.n_a * self.parameters["gamma_a"]

        v[idx("a2"), idx("a2")] = self.n_a * self.parameters["gamma_a"]
        v[idx("a3"), idx("a2")] = -self.n_a * self.parameters["gamma_a"]

        v[idx("a3"), idx("a3")] = self.n_a * self.parameters["gamma_a"]
        # No explicit transition from A3 here (absorbing into recovered class)

        # Symptomatic: I1 -> I2 -> I3 -> recovery/hospitalisation
        v[idx("i1"), idx("i1")] = self.n_i * self.parameters["gamma_s"]
        v[idx("i2"), idx("i1")] = -self.n_i * self.parameters["gamma_s"]

        v[idx("i2"), idx("i2")] = self.n_i * self.parameters["gamma_s"]
        v[idx("i3"), idx("i2")] = -self.n_i * self.parameters["gamma_s"]

        v[idx("i3"), idx("i3")] = self.n_i * self.parameters["gamma_s"]
        # Final symptomatic stage transitions out (not explicitly tracked here)

        # Store inverse for NGM computation
        self.v_inv = np.linalg.inv(v)

    def _get_f(self, contact_mtx: np.ndarray) -> np.ndarray:
        """
        Constructs the F matrix (new infection terms) for the NGM method.

        F describes how infections enter L1 due to contact with:
        - presymptomatic infectious (ip)
        - asymptomatics (a1, a2, a3)
        - symptomatics (i1, i2, i3)

        :param np.ndarray contact_mtx: Age-structured contact matrix
        :return np.ndarray: F matrix of size ((n_age * n_states) x (n_age * n_states))
        """
        i = self.i  # mapping from state -> flattened index offset
        s_mtx = self.s_mtx  # block offset for states
        n_states = self.n_states

        f = np.zeros((self.n_age * n_states, self.n_age * n_states))
        inf_a = self.parameters["inf_a"] if "inf_a" in self.parameters.keys() else 1.0
        inf_s = self.parameters["inf_s"] if "inf_s" in self.parameters.keys() else 1.0
        inf_p = self.parameters["inf_p"] if "inf_p" in self.parameters.keys() else 1.0

        susc_vec = self.parameters["susc"].reshape((-1, 1))
        f[i["l1"]:s_mtx:n_states, i["ip"]:s_mtx:n_states] = inf_p * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["a1"]:s_mtx:n_states] = inf_a * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["a2"]:s_mtx:n_states] = inf_a * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["a3"]:s_mtx:n_states] = inf_a * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["i1"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["i2"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["i3"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec

        return f

    def _get_e(self):
        """
        Builds the vector e used in the NGM calculation.
        The vector e marks the entry state for new infections (L1 in each age group).

        Structure:
            e = [1, 0, ..., 0] repeated for each age group,
        implemented as a block-diagonal stacking of unit vectors.
        """
        block = np.zeros(self.n_states)
        block[0] = 1  # L1 is the entry compartment

        self.e = block
        for i in range(1, self.n_age):
            self.e = block_diag(self.e, block)
