import numpy as np

from src.data_transformer import DataTransformer
from src.r0_generator import R0Generator


class TargetCalculator:
    def __init__(self, data_tr: DataTransformer, cm_sim: np.ndarray):
        self.data_tr = data_tr
        self.cm_sim = cm_sim
        self.output = []

    def _get_output(self, cm_sim: np.ndarray):
        r0generator = R0Generator(param=self.data_tr.params)
        beta_lhs = self.data_tr.base_r0 / r0generator.get_eig_val(
            contact_mtx=cm_sim, susceptibles=self.data_tr.susceptibles.reshape(1, -1),
            population=self.data_tr.population)[0]
        r0_lhs = (self.data_tr.beta / beta_lhs) * self.data_tr.base_r0
        output = np.array([0, r0_lhs])
        output = np.append(output, np.zeros(self.data_tr.n_ag))
        self.output = output
        return output
