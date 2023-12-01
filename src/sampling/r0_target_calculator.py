import numpy as np

import src
from src.model.r0_generator import R0Generator
from src.sampling.target_calculator import TargetCalculator
from src.seirsv.r0_seirsv import R0SeirSVModel


class R0TargetCalculator(TargetCalculator):
    def __init__(self, sim_obj: src.SimulationNPI, epi_model: str):
        self.epi_model = epi_model
        super().__init__(sim_obj=sim_obj)
        self.base_r0 = sim_obj.sim_state["base_r0"]
        self.beta = sim_obj.sim_state["beta"]

    def get_output(self, cm: np.ndarray):
        if self.epi_model == "seirSV_model":
            r0generator = R0SeirSVModel(param=self.sim_obj.params)
            beta_lhs = self.base_r0 / r0generator.get_eig_val(
                contact_mtx=cm, susceptibles=self.sim_obj.susceptibles.reshape(1, -1),
                population=self.sim_obj.population)[0]
            r0_lhs = (self.beta / beta_lhs) * self.base_r0
            output = np.array([r0_lhs])
            return output
        else:
            r0generator = R0Generator(param=self.sim_obj.params)
            beta_lhs = self.base_r0 / r0generator.get_eig_val(
                contact_mtx=cm, susceptibles=self.sim_obj.susceptibles.reshape(1, -1),
                population=self.sim_obj.population)[0]
            r0_lhs = (self.beta / beta_lhs) * self.base_r0
            output = np.array([r0_lhs])
            return output






