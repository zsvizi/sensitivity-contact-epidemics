
import numpy as np
from src.r0generator import R0Generator
from src.data_transformer import Transformer


class TargetCalculator:
    def __init__(self, sim_obj: Transformer, cm_sim: np.ndarray):
        self.sim_obj = sim_obj
        self.cm_sim = cm_sim
        self.output = []
        self._get_output(cm_sim=cm_sim)

    def _get_output(self, cm_sim: np.ndarray):
        r0generator = R0Generator(param=self.sim_obj.params)
        beta_lhs = self.sim_obj.base_r0 / r0generator.get_eig_val(
            contact_mtx=cm_sim, susceptibles=self.sim_obj.susceptibles.reshape(1, -1),
            population=self.sim_obj.population)[0]
        r0_lhs = (self.sim_obj.beta / beta_lhs) * self.sim_obj.base_r0
        output = np.array([0, r0_lhs])
        output = np.append(output, np.zeros(self.sim_obj.n_ag))
        self.output = output
        return output

