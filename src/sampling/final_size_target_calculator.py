import os
import numpy as np

from src.simulation_npi import SimulationNPI
from src.sampling.target_calculator import TargetCalculator


class FinalSizeTargetCalculator(TargetCalculator):
    def __init__(self, sim_obj: SimulationNPI):
        super().__init__(sim_obj=sim_obj)
        self.sample_params = dict()
        self.age_deaths = np.array([])

    def get_output(self, cm: np.ndarray):
        time = np.arange(0, 250, 0.5)
        solution = self.sim_obj.model.get_solution(t=time, parameters=self.sim_obj.params, cm=cm)
        # consider the total number of deaths
        idx_death = self.sim_obj.model.c_idx["d"] * self.sim_obj.n_ag     # 224:240
        age_group_deaths = solution[-1:, idx_death:(idx_death + self.sim_obj.n_ag)]
        self.age_deaths = age_group_deaths.reshape((-1, 1))
        death_final = np.sum(solution[-1:, idx_death:(idx_death + self.sim_obj.n_ag)])  # deaths at time t i.e 239:240

        # consider icu_max
        idx_icu = self.sim_obj.model.c_idx["ic"] * self.sim_obj.n_ag
        icu_max = solution[:, idx_icu:(idx_icu + self.sim_obj.n_ag)].max()  # 11,  176:192

        # consider hospitalization
        idx_h = self.sim_obj.model.c_idx["ih"] * self.sim_obj.n_ag
        h_2 = np.sum(solution[-1:, idx_h:(idx_h + self.sim_obj.n_ag)])  # hospitalization at time t
        h_1 = np.sum(solution[-2:, idx_h: (idx_h + self.sim_obj.n_ag)])   # hospitalization at time t-1
        hospitalization = abs(h_2 - h_1)

        output = np.array([0, death_final])
        output = np.append(output, np.zeros(self.sim_obj.n_ag))
        return output
