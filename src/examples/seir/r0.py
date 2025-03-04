import numpy as np
from scipy.linalg import block_diag

from src.model.r0_generator_base import R0GeneratorBase


class R0SeirSVModel(R0GeneratorBase):
    def __init__(self, param: dict, country: str = "UK", n_age: int = 15) -> None:
        self.country = country

        state = ["e", "i"]
        super().__init__(param=param, states=state, n_age=n_age)

        self._get_e()
        self._get_v()

    def _get_v(self) -> np.array:
        idx = self._idx
        v = np.zeros((self.n_age * self.n_states, self.n_age * self.n_states))
        # e -> e
        v[idx("e"), idx("e")] = self.parameters["gamma"]
        # e -> i
        v[idx("i"), idx("e")] = -self.parameters["gamma"]
        # i -> i
        v[idx("i"), idx("i")] = self.parameters["rho"]
        self.v_inv = np.linalg.inv(v)

    def _get_f(self, cm: np.array) -> np.array:
        i = self.i
        s_mtx = self.s_mtx
        n_state = self.n_states
        susc_vec = self.parameters["susc"].reshape((-1, 1))
        f = np.zeros((self.n_age * n_state, self.n_age * n_state))
        f[i["e"]:s_mtx:n_state, i["e"]:s_mtx:n_state] = cm.T * susc_vec
        f[i["i"]:s_mtx:n_state, i["i"]:s_mtx:n_state] = cm.T * susc_vec
        return f

    def _get_e(self):
        block = np.zeros(self.n_states, )
        block[0] = 1
        self.e = block
        for i in range(1, self.n_age):
            self.e = block_diag(self.e, block)
