import numpy as np

import src
from src.examples.rost.r0 import R0Generator
from src.sampling.target.target_calculator import TargetCalculator
from src.examples.seir.r0 import R0SeirSVModel
from src.examples.chikina.r0 import R0SirModel
from src.examples.moghadas.r0 import R0SeyedModel


class R0TargetCalculator(TargetCalculator):
    def __init__(self, sim_obj: src.SimulationNPI, country: str):
        self.country = country
        super().__init__(sim_obj=sim_obj)
        self.base_r0 = sim_obj.sim_state["base_r0"]
        self.beta = sim_obj.sim_state["beta"]

    def get_output(self, cm: np.ndarray):
        if self.country == "Hungary":
            r0generator = R0Generator(param=self.sim_obj.params)
        elif self.country == "UK":
            r0generator = R0SeirSVModel(param=self.sim_obj.params)
        elif self.country == "usa":
            r0generator = R0SirModel(param=self.sim_obj.params)
        elif self.country == "united_states":
            r0generator = R0SeyedModel(param=self.sim_obj.params)
        else:
            raise Exception("Invalid country!")

        beta_lhs = self.base_r0 / r0generator.get_eig_val(
            contact_mtx=cm, susceptibles=self.sim_obj.susceptibles.reshape(1, -1),
            population=self.sim_obj.population)[0]
        r0_lhs = (self.beta / beta_lhs) * self.base_r0
        output = np.array([r0_lhs])
        return output
