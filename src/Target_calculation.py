import numpy as np


class TargetCalculator:
    def __init__(self, cm_sim, sim_state: dict, sim_obj):
        self.cm_sim = cm_sim
        self.sim_obj = sim_obj
        self.base_r0 = sim_state["base_r0"]
        self.beta = sim_state["beta"]
        self.r0generator = sim_state["r0generator"]

    def get_output(self, cm_sim: np.ndarray):
        beta_lhs = self.base_r0 / self.r0generator.get_eig_val(contact_mtx=cm_sim,
                                                               susceptibles=self.sim_obj.susceptibles.reshape(1, -1),
                                                               population=self.sim_obj.population)[0]
        r0_lhs = (self.beta / beta_lhs) * self.base_r0
        output = np.array([0, r0_lhs])
        output = np.append(output, np.zeros(self.sim_obj.no_ag))
        return output
