import numpy as np

from src.simulation_npi import SimulationNPI
from src.sampling.target_calculator import TargetCalculator


class FinalSizeTargetCalculator(TargetCalculator):
    def __init__(self, sim_obj: SimulationNPI):
        super().__init__(sim_obj=sim_obj)

        self.age_hospitalized = np.array([])
        self.sample_params = dict()
        self.age_deaths = np.array([])
        self.total_infectious = np.array([])

    def get_output(self, cm: np.ndarray):
        time = np.arange(0, 750, 0.5)
        solution = self.sim_obj.model.get_solution(t=time, parameters=self.sim_obj.params, cm=cm)

        # sum of infectious persons, that is all the compartments excluding s, r, & d
        no_infected = self.sim_obj.model.aggregate_by_age(solution=solution,
                                                         idx=self.sim_obj.model.c_idx["c"] -
                                                             (self.sim_obj.model.c_idx["s"] +
                                                              self.sim_obj.model.c_idx["d"] +
                                                              self.sim_obj.model.c_idx["r"]))
        # Loop infinitely
        while True:
            # consider the total number of hospitalized
            age_group_hospitalized = self.sim_obj.model.aggregate_by_age(
                solution=solution, idx=self.sim_obj.model.c_idx["ih"])
            age_hospitalized = age_group_hospitalized / np.sum(age_group_hospitalized)
            self.age_hospitalized = age_hospitalized

            # consider the total number of deaths
            age_group_deaths = self.sim_obj.model.aggregate_by_age(
                solution=solution, idx=self.sim_obj.model.c_idx["d"])
            age_deaths = age_group_deaths / np.sum(age_group_deaths)
            self.age_deaths = age_deaths.reshape((-1, 1))

            death_final = np.sum(age_group_deaths)
            output = np.array([death_final])
            if np.sum(no_infected) < 1:
                # If so, exit the loop
                break
        return output
