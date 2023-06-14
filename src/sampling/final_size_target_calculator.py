import numpy as np

from src.simulation_npi import SimulationNPI
from src.sampling.target_calculator import TargetCalculator


class FinalSizeTargetCalculator(TargetCalculator):
    def __init__(self, sim_obj: SimulationNPI):
        super().__init__(sim_obj=sim_obj)
        self.hospitalized = np.array([])
        self.final_deaths = np.array([])
        self.age_hospitalized = np.array([])
        self.sample_params = dict()
        self.age_deaths = np.array([])

    def get_output(self, cm: np.ndarray):
        time = np.arange(0, 250, 0.5)
        solution = self.sim_obj.model.get_solution(t=time, parameters=self.sim_obj.params, cm=cm)
        # consider the total number of hospitalized
        idx_h = self.sim_obj.model.c_idx["ih"] * self.sim_obj.n_ag  # 160:176
        age_group_hospitalized = np.sum(solution[:, idx_h:(idx_h + self.sim_obj.n_ag)], axis=0)
        age_hospitalized = age_group_hospitalized / np.sum(age_group_hospitalized)
        self.hospitalized = age_group_hospitalized
        self.age_hospitalized = age_hospitalized

        # consider the total number of deaths
        idx_death = self.sim_obj.model.c_idx["d"] * self.sim_obj.n_ag     # 224:240
        age_group_deaths = solution[-1:, idx_death:(idx_death + self.sim_obj.n_ag)]
        age_deaths = age_group_deaths / np.sum(age_group_deaths)
        self.final_deaths = age_group_deaths.reshape((-1, 1))
        self.age_deaths = age_deaths.reshape((-1, 1))
        death_final = np.sum(age_group_deaths)

        output = np.array([death_final])
        return output
