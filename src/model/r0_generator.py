import numpy as np
from scipy.linalg import block_diag

from src.model.r0 import R0GeneratorBase


class R0Generator(R0GeneratorBase):
    def __init__(self, param: dict, n_age: int = 16) -> None:
        states = ["l1", "l2", "ip", "a1", "a2", "a3", "i1", "i2", "i3"]
        super().__init__(param=param, states=states, n_age=n_age)
        self.n_l = 2
        self.n_a = 3
        self.n_i = 3

        self._get_e()
        self._get_v()

    def _get_v(self) -> np.array:
        idx = self._idx
        v = np.zeros((self.n_age * self.n_states, self.n_age * self.n_states))
        # L1 -> L2
        v[idx("l1"), idx("l1")] = self.n_l * self.parameters["alpha_l"]
        v[idx("l2"), idx("l1")] = -self.n_l * self.parameters["alpha_l"]
        # L2 -> Ip
        v[idx("l2"), idx("l2")] = self.n_l * self.parameters["alpha_l"]
        v[idx("ip"), idx("l2")] = -self.n_l * self.parameters["alpha_l"]
        # ip -> I1/A1
        v[idx("ip"), idx("ip")] = self.parameters["alpha_p"]
        v[idx("i1"), idx("ip")] = -self.parameters["alpha_p"] * (1 - self.parameters["p"])
        v[idx("a1"), idx("ip")] = -self.parameters["alpha_p"] * self.parameters["p"]
        # A1 -> A2
        v[idx("a1"), idx("a1")] = self.n_a * self.parameters["gamma_a"]
        v[idx("a2"), idx("a1")] = -self.n_a * self.parameters["gamma_a"]
        # A2 -> A3
        v[idx("a2"), idx("a2")] = self.n_a * self.parameters["gamma_a"]
        v[idx("a3"), idx("a2")] = -self.n_a * self.parameters["gamma_a"]
        # A3 -> R
        v[idx("a3"), idx("a3")] = self.n_a * self.parameters["gamma_a"]
        # I1 -> I2
        v[idx("i1"), idx("i1")] = self.n_i * self.parameters["gamma_s"]
        v[idx("i2"), idx("i1")] = -self.n_i * self.parameters["gamma_s"]
        # I2 -> I3
        v[idx("i2"), idx("i2")] = self.n_i * self.parameters["gamma_s"]
        v[idx("i3"), idx("i2")] = -self.n_i * self.parameters["gamma_s"]
        # I3 -> Ih/Ic (& R)
        v[idx("i3"), idx("i3")] = self.n_i * self.parameters["gamma_s"]

        self.v_inv = np.linalg.inv(v)

    def _get_f(self, contact_mtx: np.array) -> np.array:
        i = self.i
        s_mtx = self.s_mtx
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
        block = np.zeros(self.n_states, )
        block[0] = 1
        self.e = block
        for i in range(1, self.n_age):
            self.e = block_diag(self.e, block)
