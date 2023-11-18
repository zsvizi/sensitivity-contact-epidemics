import numpy as np
import torch


def get_n_states(n_classes, comp_name):
    return [f"{comp_name}_{i}" for i in range(n_classes)]


def generate_transition_block(transition_param: float, n_states: int) -> torch.Tensor:
    trans_block = torch.zeros((n_states, n_states))
    # Outflow from states (diagonal elements)
    trans_block = trans_block.fill_diagonal_(-transition_param)
    # Inflow to states (elements under the diagonal)
    trans_block[1:, :n_states - 1] = trans_block[1:, :n_states - 1].fill_diagonal_(transition_param)
    return trans_block


def generate_transition_matrix(trans_param_dict, ps, n_age, n_comp, c_idx):
    trans_matrix = torch.zeros((n_age * n_comp, n_age * n_comp))
    for age_group in range(n_age):
        for comp, trans_param in trans_param_dict.items():
            n_states = ps[f'n_{comp}']
            diag_idx = age_group * n_comp + c_idx[f'{comp}_0']
            block_slice = slice(diag_idx, diag_idx + n_states)
            # Create transition block from each transitional state
            trans_matrix[block_slice, block_slice] = generate_transition_block(trans_param, n_states)
    return trans_matrix


class MatrixGenerator:
    def __init__(self, model, cm, ps: dict, xs: np.ndarray):
        self.xs = xs
        self.cm = cm
        self.ps = ps
        self.s_mtx = model.n_age * model.n_comp
        self.n_state_comp = model.n_state_comp
        self.n_age = model.n_age
        self.n_comp = model.n_comp
        self.population = model.population
        self.device = model.device
        self.idx = model.idx
        self.c_idx = model.c_idx
        self._get_trans_param_dict()

    def get_A(self):
        # Multiplied with y, the resulting 1D tensor contains the rate of transmission for the susceptibles of
        # age group i at the indices of compartments s^i and e_0^i
        A = torch.zeros((self.s_mtx, self.s_mtx)).to(self.device)
        s, l1, l2, ip, ia1, ia2, ia3, is1, is2, is3, ih, ic, icr, r, d, c = self.xs.reshape(-1, self.n_age)
        transmission_rate = self.ps["beta"] * np.array((ip + self.ps["inf_a"] * (ia1 + ia2 + ia3) +
                                                   (is1 + is2 + is3))).dot(self.cm)
        idx = self.idx
        A[idx('s'), idx('s')] = - transmission_rate
        A[idx('l1'), idx('s')] = transmission_rate
        return A

    def get_T(self):
        T = torch.zeros((self.s_mtx, self.s_mtx)).to(self.device)
        # Multiplied with y, the resulting 1D tensor contains the sum of all contacts with infecteds of
        # age group i at indices of compartments s_i and e_i^0
        for i_state in get_n_states(self.ps["n_l1"], "l1"):
            T[self._get_comp_slice('s'), self._get_comp_slice(i_state)] = self.cm.T
            T[self._get_comp_slice('l1_0'), self._get_comp_slice(i_state)] = self.cm.T
            T[self._get_comp_slice('l2_0'), self._get_comp_slice(i_state)] = self.cm.T
            T[self._get_comp_slice('1p_0'), self._get_comp_slice(i_state)] = self.cm.T
            T[self._get_comp_slice('a1_0'), self._get_comp_slice(i_state)] = self.cm.T
            T[self._get_comp_slice('a2_0'), self._get_comp_slice(i_state)] = self.cm.T
            T[self._get_comp_slice('a3_0'), self._get_comp_slice(i_state)] = self.cm.T
            T[self._get_comp_slice('i1_0'), self._get_comp_slice(i_state)] = self.cm.T
            T[self._get_comp_slice('i2_0'), self._get_comp_slice(i_state)] = self.cm.T
            T[self._get_comp_slice('i3_0'), self._get_comp_slice(i_state)] = self.cm.T

        return T

    def get_B(self):
        ps = self.ps
        # B is the tensor representing the first order elements of the ODE system. We begin with filling in
        # the transition blocks of the erlang distributed parameters
        B = generate_transition_matrix(self.trans_param_dict, self.ps, self.n_age, self.n_comp, self.c_idx)
        # Then do the rest of the first order terms
        idx = self.idx
        c_end = self._get_end_state
        l1_end = c_end('l1')
        l2_end = c_end('l2')
        ip_end = c_end('ip')
        a1_end = c_end('a1')
        a2_end = c_end('a2')
        a3_end = c_end('a3')
        i1_end = c_end('i1')
        i2_end = c_end('i2')
        i3_end = c_end('i3')
        h_end = c_end('h')
        ic_end = c_end('ic')
        icr_end = c_end('icr')
        c_end = c_end('c')

        # L1 -> L2
        B[idx("l2_0"), idx(l1_end)] = 2 * self.ps["alpha_l"]
        # L2 -> Ip
        B[idx("ip"), idx(l2_end)] = 2 * self.ps["alpha_l"]
        # ip -> A1
        B[idx("a1"), idx(ip_end)] = self.ps["alpha_p"] * self.ps["p"]
        # A1 -> A2
        B[idx("a2"), idx(a1_end)] = 3 * self.ps["gamma_a"]
        # A2 -> A3
        B[idx("a3"), idx(a2_end)] = 3 * self.ps["gamma_a"]
        # ip -> i1
        B[idx("i1"), idx(ip_end)] = self.ps["alpha_p"] * (1 - self.ps["p"])
        # i1 -> i2
        B[idx("i2"), idx(i1_end)] = 3 * self.ps["gamma_s"]
        # i2 -> i3
        B[idx("i3"), idx(i2_end)] = 3 * self.ps["gamma_s"]
        # A3 -> R
        B[idx("r"), idx(a3_end)] = 3 * self.ps["gamma_a"]
        # i3 -> R
        B[idx("r"), idx(i3_end)] = 3 * self.ps["gamma_s"] * (1 - self.ps["h"])
        # i3 -> h
        B[idx("h_0"), idx(i3_end)] = 3 * self.ps["gamma_s"] * self.ps["h"]
        # i   ->  IC
        B[idx('ic_0'), idx(i3_end)] = ps["xi"] * ps["h"] * 3 * ps["gamma_s"]
        # H   ->  R
        B[idx('r'), idx(h_end)] = 1 - ps["h"] * 3 * ps["gamma_s"]
        # IC  ->  ICR
        B[idx('icr_0'), idx(ic_end)] = ps["gamma_c"] * (1 - ps["mu"])
        # ICR ->  R
        B[idx('r'), idx(icr_end)] = ps["gamma_cr"]
        # IC  ->  D
        B[idx('d'), idx(ic_end)] = ps["gamma_c"] * ps["mu"]
        # c   ->  i
        B[idx("s"), idx(c_end)] = 2 * ps["alpha_l"]
        return B

    def _get_comp_slice(self, comp: str) -> slice:
        return slice(self.c_idx[comp], self.s_mtx, self.n_comp)

    def _get_end_state(self, comp: str) -> str:
        n_states = self.ps[f'n_{comp}']
        return f'{comp}_{n_states - 1}'

    def _get_trans_param_dict(self):
        ps = self.ps
        trans_param_list = [ps["alpha_l"], ps["alpha_p"], ps["gamma_a"], ps["gamma_s"], ps["gamma_h"],
        ps["gamma_c"], ps["gamma_cr"]]
        self.trans_param_dict = {key: value for key, value in zip(self.n_state_comp, trans_param_list)}
