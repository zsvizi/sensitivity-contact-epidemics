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
        t_interval = 100
        t = np.arange(0, t_interval, 0.5)
        # solve the model from 0 to 100 using the initial condition
        sol = self.sim_obj.model.get_solution(
            init_values=self.sim_obj.model.get_initial_values,
            t=t,
            parameters=self.sim_obj.params,
            cm=cm)

        state = sol[-1]

        # introduce a variable for tracking how long the ODE was solved
        t_interval_complete = 0
        t_interval_complete += t_interval

        hospital_peak = (self.sim_obj.model.aggregate_by_age(
                solution=sol, idx=self.sim_obj.model.c_idx["ih"]) +
                                 self.sim_obj.model.aggregate_by_age(
                solution=sol, idx=self.sim_obj.model.c_idx["ic"]) +
                                 self.sim_obj.model.aggregate_by_age(
                solution=sol, idx=self.sim_obj.model.c_idx["icr"])).max()

        # Loop infinitely
        while True:
            hospital_peak_now = (self.sim_obj.model.aggregate_by_age(
                solution=sol, idx=self.sim_obj.model.c_idx["ih"])[-1] +
                                 self.sim_obj.model.aggregate_by_age(
                solution=sol, idx=self.sim_obj.model.c_idx["ic"])[-1] +
                                 self.sim_obj.model.aggregate_by_age(
                solution=sol, idx=self.sim_obj.model.c_idx["icr"])[-1]).max()

            # check whether it is higher than it was in the previous turn
            if hospital_peak_now > hospital_peak:
                hospital_peak = hospital_peak_now

            n_infecteds = self.sim_obj.model.aggregate_by_age(
                solution=sol, idx=self.sim_obj.model.c_idx["c"])[-1] + \
                          (self.sim_obj.model.aggregate_by_age(
                              solution=sol, idx=self.sim_obj.model.c_idx["l1"])[-1] +
                           self.sim_obj.model.aggregate_by_age(
                               solution=sol, idx=self.sim_obj.model.c_idx["l2"])[-1] +
                           self.sim_obj.model.aggregate_by_age(
                               solution=sol, idx=self.sim_obj.model.c_idx["ih"])[-1] +
                           self.sim_obj.model.aggregate_by_age(
                               solution=sol, idx=self.sim_obj.model.c_idx["ic"])[-1] +
                           self.sim_obj.model.aggregate_by_age(
                               solution=sol, idx=self.sim_obj.model.c_idx["icr"])[-1] -
                           self.sim_obj.model.aggregate_by_age(
                               solution=sol, idx=self.sim_obj.model.c_idx["d"])[-1]
                           )

            # check whether the previously calculated number is less than 1
            if n_infecteds < 1:
                break
            # since the number of infecteds is above 1, we solve the ODE again
            # from 0 to t_interval using the current state
            sol = self.sim_obj.model.get_solution(
                init_values=state,
                t=t,
                parameters=self.sim_obj.params,
                cm=cm)
            state = sol[-1]
            t_interval_complete += t_interval
        sol = sol
        final_size_dead = np.sum(self.sim_obj.model.aggregate_by_age(
            solution=sol,
            idx=self.sim_obj.model.c_idx["d"])[-1])
        output = np.array([final_size_dead])
        return output
