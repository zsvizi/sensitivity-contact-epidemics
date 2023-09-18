import numpy as np
from collections import Counter
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
        t = 100
        time = np.arange(0, t, 0.5)
        # solve the model from 0 to 100 starting from the vector at last time point
        solution = self.sim_obj.model.get_solution(
            t=time,
            init_values=self.sim_obj.model.get_initial_values()[-1],
            parameters=self.sim_obj.params, cm=cm)

        # solve the model from 100-600 using initial condition from the vector at the last time step
        t2 = 600
        time_2 = np.arange(100, t2, 0.5)
        sol = self.sim_obj.model.get_solution(t=time_2,
                                              init_values=solution[-1],
                                              parameters=self.sim_obj.params, cm=cm)

        # Loop infinitely
        while True:
            # consider the total number of hospitalized
            age_group_hospitalized = self.sim_obj.model.aggregate_by_age(
                solution=sol, idx=self.sim_obj.model.c_idx["ih"])

            # hospitalized at the last time step for each age group
            age_group_hospitalized = age_group_hospitalized[-1]
            age_hospitalized = age_group_hospitalized / np.sum(age_group_hospitalized)
            self.age_hospitalized = age_hospitalized

            # consider the total number of deaths
            age_group_deaths = self.sim_obj.model.aggregate_by_age(
                solution=sol, idx=self.sim_obj.model.c_idx["d"])
            # number of deaths at the last time step for each age group
            age_group_deaths = age_group_deaths[-1]
            age_deaths = age_group_deaths / np.sum(age_group_deaths)
            self.age_deaths = age_deaths.reshape((-1, 1))
            # total deaths
            death_final = np.sum(age_group_deaths)
            output = np.array([death_final])

            # sum of infectious persons, that is all the compartments excluding s, r, & d
            no_infected = self.sim_obj.model.aggregate_by_age(solution=sol,
                                                              idx=self.sim_obj.model.c_idx["c"] -
                                                                  (self.sim_obj.model.c_idx["s"] +
                                                                   self.sim_obj.model.c_idx["d"] +
                                                                   self.sim_obj.model.c_idx["r"]))
            # total number of infected at the last time step
            n_infecteds = np.sum(no_infected[-1])

            # count the number of times the model is solved using counter from collections
            model_count = Counter(sol[t2])
            n_model_counts = model_count.values()
            if n_infecteds < 1:
                break
        return output








