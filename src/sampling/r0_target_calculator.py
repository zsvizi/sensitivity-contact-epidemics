import numpy as np

import src
from src.model.r0_generator import R0Generator
from src.sampling.target_calculator import TargetCalculator


class R0TargetCalculator(TargetCalculator):
    def __init__(self, sim_state: dict, sim_obj: src.SimulationNPI):
        super().__init__(sim_obj=sim_obj)
        self.base_r0 = sim_state["base_r0"]
        self.beta = sim_state["beta"]

    def get_output(self, cm: np.ndarray):
        r0generator = R0Generator(param=self.sim_obj.params)
        beta_lhs = self.base_r0 / r0generator.get_eig_val(
            contact_mtx=cm, susceptibles=self.sim_obj.susceptibles.reshape(1, -1),
            population=self.sim_obj.population)[0]
        r0_lhs = (self.beta / beta_lhs) * self.base_r0

        output = np.array([0, r0_lhs])
        output = np.append(output, np.zeros(self.sim_obj.n_ag))
        return output
