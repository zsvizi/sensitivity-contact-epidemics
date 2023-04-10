import numpy as np
from src.simulation_npi import SimulationNPI
from src.model.r0_generator import R0Generator


class TargetCalculator:
    def __init__(self, sim_state: dict, sim_obj: SimulationNPI):
        self.sim_obj = sim_obj
        self.base_r0 = sim_state["base_r0"]
        self.beta = sim_state["beta"]

    def get_output(self, cm_sim: np.ndarray):
        r0generator = R0Generator(param=self.sim_obj.params)
        beta_lhs = self.base_r0 / r0generator.get_eig_val(
            contact_mtx=cm_sim, susceptibles=self.sim_obj.susceptibles.reshape(1, -1),
            population=self.sim_obj.population)[0]
        r0_lhs = (self.beta / beta_lhs) * self.base_r0
        output = np.array([0, r0_lhs])
        output = np.append(output, np.zeros(self.sim_obj.n_ag))
        return output
