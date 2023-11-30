import numpy as np
from scipy.linalg import block_diag

from src.model.r0 import R0GeneratorBase


class R0SeirSVModel(R0GeneratorBase):
    def __init__(self, uk_param: dict, uk_cm: np.ndarray, n_age: int = 16) -> None:
        state = ["e", "i"]
        self.uk_param = uk_param
        self.uk_cm = uk_cm
        super().__init__(param=uk_param, states=state, n_age=n_age)

        self._get_e()
        self._get_v()

    def _get_v(self) -> np.array:
        idx = self._idx
        v = np.zeros((self.n_age * self.n_states, self.n_age * self.n_states))
        v[idx("e"), idx("e")] = self.uk_param["gamma"]
        v[idx("i"), idx("i")] = self.uk_param["rho"]
        self.v_inv = np.linalg.inv(v)

    def _get_f(self, cm: np.array) -> np.array:
        i = self.i
        s_mtx = self.s_mtx
        n_state = self.n_states
        f = np.zeros((self.n_age * n_state, self.n_age * n_state))
        f[i["e"]:s_mtx:n_state, i["e"]:s_mtx:n_state] = self.uk_cm.T
        f[i["i"]:s_mtx:n_state, i["i"]:s_mtx:n_state] = self.uk_cm.T
        return f

    def _get_e(self):
        block = np.zeros(self.n_states, )
        block[0] = 1
        self.e = block
        for i in range(1, self.n_age):
            self.e = block_diag(self.e, block)
