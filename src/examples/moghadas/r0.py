import numpy as np
from scipy.linalg import block_diag

from src.model.r0_generator_base import R0GeneratorBase


class R0SeyedModel(R0GeneratorBase):
    def __init__(self, param: dict, n_age: int = 4,
                 country: str = "united_states") -> None:
        self.country = country

        states = ["e", "i_n", "q_n", "i_h", "q_h", "a_n", "a_q"]
        super().__init__(param=param, states=states, n_age=n_age)
        self._get_e()
        self._get_v()

    def _get_v(self) -> np.array:
        idx = self._idx
        ps = self.parameters
        v = np.zeros((self.n_age * self.n_states, self.n_age * self.n_states))
        # e -> e
        v[idx("e"), idx("e")] = ps["sigma"]
        # e -> i_n
        v[idx("i_n"), idx("e")] = -(1 - ps["theta"]) * (1 - ps["q"]) * (1 - ps["h"]) * ps["sigma"]
        # i_n -> i_n
        v[idx("i_n"), idx("i_n")] = (1 - ps["f_i"]) * ps["gamma"] - ps["f_i"] * ps["tau_i"]
        # e -> q_n
        v[idx("q_n"), idx("e")] = -(1 - ps["theta"]) * ps["q"] * (1 - ps["h"]) * ps["sigma"]
        # i_n -> q_n
        v[idx("q_n"), idx("i_n")] = -ps["f_i"] * ps["tau_i"]
        # q_n -> q_n
        v[idx("q_n"), idx("q_n")] = ps["gamma"]
        # e -> i_h
        v[idx("i_h"), idx("e")] = -(1 - ps["theta"]) * (1 - ps["q"]) * ps["h"] * ps["sigma"]
        # i_h -> i_h
        v[idx("i_h"), idx("i_h")] = (1 - ps["f_i"]) * ps["delta"] - ps["f_i"] * ps["tau_i"]
        # e -> q_h
        v[idx("q_h"), idx("e")] = -(1 - ps["theta"]) * ps["q"] * ps["h"] * ps["sigma"]
        # q_h -> q_h
        v[idx("q_h"), idx("q_h")] = ps["delta"]
        # i_h -> q_h
        v[idx("q_h"), idx("i_h")] = -ps["f_i"] * ps["tau_i"]
        # e -> a_n
        v[idx("a_n"), idx("e")] = -ps["theta"] * ps["sigma"]
        # a_n -> a_n
        v[idx("a_n"), idx("a_n")] = (1 - ps["f_a"]) * ps["gamma"] - ps["f_a"] * ps["tau_a"]
        # a_n -> a_q
        v[idx("a_q"), idx("a_n")] = -ps["f_a"] * ps["tau_a"]
        # a_q -> a_q
        v[idx("a_q"), idx("a_q")] = ps["gamma"]

        self.v_inv = np.linalg.inv(v)

    def _get_f(self, contact_mtx: np.array) -> np.array:
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states
        susc_vec = self.parameters["susc"].reshape((-1, 1))

        f = np.zeros((self.n_age * n_states, self.n_age * n_states))
        inf_a = self.parameters["k"] if "k" in self.parameters.keys() else 1.0
        inf_s = self.parameters["inf_s"] if "inf_s" in self.parameters.keys() else 1.0

        f[i["e"]:s_mtx:n_states, i["a_n"]:s_mtx:n_states] = inf_a * contact_mtx.T * susc_vec
        f[i["e"]:s_mtx:n_states, i["a_q"]:s_mtx:n_states] = inf_a * contact_mtx.T * susc_vec
        f[i["e"]:s_mtx:n_states, i["i_n"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec
        f[i["e"]:s_mtx:n_states, i["i_h"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec
        f[i["e"]:s_mtx:n_states, i["q_n"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec
        f[i["e"]:s_mtx:n_states, i["q_h"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec

        return f

    def _get_e(self):
        block = np.zeros(self.n_states, )
        block[0] = 1
        self.e = block
        for i in range(1, self.n_age):
            self.e = block_diag(self.e, block)
