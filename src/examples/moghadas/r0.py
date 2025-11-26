import numpy as np
from scipy.linalg import block_diag

from src.model.r0_generator_base import R0GeneratorBase


class R0SeyedModel(R0GeneratorBase):
    """
    Age-structured infectious disease model following a Moghadas-type
    compartmental structure.

    States per age group:

        e     : exposed (latent, not yet infectious)
        i_n   : infectious, non-isolated (mild/typical community infection)
        q_n   : isolated/quarantined non-severe infection
        i_h   : infectious requiring hospitalization
        q_h   : isolated/quarantined hospitalized infection
        a_n   : asymptomatic infectious in the community
        a_q   : asymptomatic infectious in isolation
    """

    def __init__(self, param: dict, n_age: int = 4,
                 country: str = "united_states") -> None:
        """
        :param dict param: Parameter dictionary containing all epidemiological parameters
         (transition rates, probabilities, susceptibility, infectiousness multipliers, etc.)
        :param int n_age: Number of age groups
        :param str country: Country identifier for loading parameter presets
        """
        self.country = country

        # Order of compartments inside each age block.
        states = ["e", "i_n", "q_n", "i_h", "q_h", "a_n", "a_q"]

        super().__init__(param=param, states=states, n_age=n_age)

        self._get_e()
        self._get_v()

    def _get_v(self) -> None:
        """
        Constructs the V matrix (transitions out of infected compartments)
        for the next-generation matrix approach. After construction, V is inverted and stored as self.v_inv (np.ndarray)

        V matrix:
        - diagonal: rate of leaving each compartment
        - off-diagonal: negative transition rates into next-stage compartments
        """
        idx = self._idx
        ps = self.parameters

        v = np.zeros((self.n_age * self.n_states,
                      self.n_age * self.n_states))

        # e -> e
        v[idx("e"), idx("e")] = ps["sigma"]

        # e -> i_n   (community symptomatic)
        v[idx("i_n"), idx("e")] = -(1 - ps["theta"]) * (1 - ps["q"]) * (1 - ps["h"]) * ps["sigma"]

        # e -> q_n   (isolated mild symptomatic)
        v[idx("q_n"), idx("e")] = -(1 - ps["theta"]) * ps["q"] * (1 - ps["h"]) * ps["sigma"]

        # e -> i_h   (hospitalized symptomatic)
        v[idx("i_h"), idx("e")] = -(1 - ps["theta"]) * (1 - ps["q"]) * ps["h"] * ps["sigma"]

        # e -> q_h   (isolated hospital cases)
        v[idx("q_h"), idx("e")] = -(1 - ps["theta"]) * ps["q"] * ps["h"] * ps["sigma"]

        # e -> a_n   (community asymptomatic)
        v[idx("a_n"), idx("e")] = -ps["theta"] * ps["sigma"]

        # i_n -> i_n
        v[idx("i_n"), idx("i_n")] = (1 - ps["f_i"]) * ps["gamma"] - ps["f_i"] * ps["tau_i"]

        # i_n -> q_n (detected -> isolation)
        v[idx("q_n"), idx("i_n")] = -ps["f_i"] * ps["tau_i"]

        # Quarantined non-severe
        v[idx("q_n"), idx("q_n")] = ps["gamma"]

        # Hospital infectious
        # i_h -> i_h
        v[idx("i_h"), idx("i_h")] = (1 - ps["f_i"]) * ps["delta"] - ps["f_i"] * ps["tau_i"]

        # i_h -> q_h (detected -> isolation)
        v[idx("q_h"), idx("i_h")] = -ps["f_i"] * ps["tau_i"]

        # Quarantined hospital
        v[idx("q_h"), idx("q_h")] = ps["delta"]

        # Asymptomatic dynamics
        # a_n -> a_n
        v[idx("a_n"), idx("a_n")] = (1 - ps["f_a"]) * ps["gamma"] - ps["f_a"] * ps["tau_a"]

        # a_n -> a_q (detected)
        v[idx("a_q"), idx("a_n")] = -ps["f_a"] * ps["tau_a"]

        # a_q -> a_q
        v[idx("a_q"), idx("a_q")] = ps["gamma"]

        # Store inverse for NGM:
        self.v_inv = np.linalg.inv(v)

    def _get_f(self, contact_mtx: np.ndarray) -> np.ndarray:
        """
        Constructs the F matrix representing *new infection terms* for the NGM.

        New infections enter the **exposed** compartment (e) in each age group.

        Infectious sources:
            - a_n, a_q (asymptomatic)
            - i_n, i_h, q_n, q_h (symptomatic & isolated)
        Each is multiplied by an infectiousness factor (k or inf_s).

        :param np.ndarray contact_mtx: Age-structured contact matrix (contacts from j to i)

        :return np.ndarray: F matrix of size (n_age*n_states x n_age*n_states)
        """
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states

        f = np.zeros((self.n_age * n_states, self.n_age * n_states))

        # Relative infectiousness multipliers (defaults = 1.0)
        inf_a = self.parameters.get("k", 1.0)
        inf_s = self.parameters.get("inf_s", 1.0)

        # Susceptibility as column vector
        susc_vec = self.parameters["susc"].reshape((-1, 1))

        # All new infections enter at the exposed compartment "e"
        # Contact matrix is transposed so F_ij = infections in i caused by j.
        for inf_state, mult in [
            ("a_n", inf_a),
            ("a_q", inf_a),
            ("i_n", inf_s),
            ("i_h", inf_s),
            ("q_n", inf_s),
            ("q_h", inf_s),
        ]:
            f[i["e"]:s_mtx:n_states, i[inf_state]:s_mtx:n_states] = (
                mult * contact_mtx.T * susc_vec
            )

        return f

    def _get_e(self):
        """
        Builds the e vector for the NGM system.

        e marks the entry compartments for new infections.
        In this model, all new infections enter in the exposed (e) state.

        Structure:
            e = [1, 0, ..., 0] repeated for each age group,
        implemented as a block-diagonal stacking of unit vectors.
        """
        block = np.zeros(self.n_states)
        block[0] = 1  # exposed ("e") is the entry compartment

        self.e = block
        for _ in range(1, self.n_age):
            self.e = block_diag(self.e, block)
