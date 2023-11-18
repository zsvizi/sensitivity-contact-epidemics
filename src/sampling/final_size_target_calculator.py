import numpy as np
from src.simulation_npi import SimulationNPI
from src.sampling.target_calculator import TargetCalculator


class FinalSizeTargetCalculator(TargetCalculator):
    def __init__(self, sim_obj: SimulationNPI, epi_model: str = "rost_model"):
        self.epi_model = epi_model
        super().__init__(sim_obj=sim_obj)

    def get_output(self, cm: np.ndarray):
        t_interval = 10
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

        if self.epi_model == "rost_model":
            # let's say we want to store the peak size of hospitalized people during the simulation
            # aggregate age groups for ih, ic and icr from the previously calculated solution
            # then get maximal value of the time series
            hospital_peak = (
                    self.sim_obj.model.aggregate_by_age(
                        solution=sol, idx=self.sim_obj.model.c_idx["ih"]) +
                    self.sim_obj.model.aggregate_by_age(
                        solution=sol, idx=self.sim_obj.model.c_idx["ic"]) +
                    self.sim_obj.model.aggregate_by_age(
                        solution=sol, idx=self.sim_obj.model.c_idx["icr"])
            ).max()
        else:
            hospital_peak = None

        # Loop infinitely
        while True:
            if self.epi_model == "rost_model":
                # let's say we want to store the peak size of hospitalized people during the simulation
                # aggregate age groups for ih, ic and icr from the previously calculated solution
                # then get maximal value of the time series
                hospital_peak_now = (
                    self.sim_obj.model.aggregate_by_age(
                        solution=sol, idx=self.sim_obj.model.c_idx["ih"]) +
                    self.sim_obj.model.aggregate_by_age(
                        solution=sol, idx=self.sim_obj.model.c_idx["ic"]) +
                    self.sim_obj.model.aggregate_by_age(
                        solution=sol, idx=self.sim_obj.model.c_idx["icr"])
                ).max()

                # check whether it is higher than it was in the previous turn
                if hospital_peak_now > hospital_peak:
                    hospital_peak = hospital_peak_now

            # calculate the number of infected individuals at the current state
            # sum of aggregation along all age groups in
            # l1, l2, ip, ia1, ia2, ia3, is1, is2, is3, ih, ic, icr

            # convert the current state to a numpy array
            if self.epi_model == "rost_model":
                state = np.array([state])
                n_infecteds = (state.sum() -
                               self.sim_obj.model.aggregate_by_age(
                                   solution=state, idx=self.sim_obj.model.c_idx["s"]) -
                               self.sim_obj.model.aggregate_by_age(
                                   solution=state, idx=self.sim_obj.model.c_idx["r"]) -
                               self.sim_obj.model.aggregate_by_age(
                                   solution=state, idx=self.sim_obj.model.c_idx["d"]) -
                               self.sim_obj.model.aggregate_by_age(
                                   solution=state, idx=self.sim_obj.model.c_idx["c"])
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

            if self.epi_model == "rost_model":
                # for calculating final size value for compartment D
                # aggregate the current state (calculated in the last turn of the loop) for compartment "d"
                final_size_dead = self.sim_obj.model.aggregate_by_age(
                    solution=np.array([state]),
                    idx=self.sim_obj.model.c_idx["d"])

                # concatenate all output values to get the array of return
                output = np.array([final_size_dead])
                return output
            elif self.epi_model == "sir_model":
                final_size = self.sim_obj.model.aggregate_by_age(
                    solution=np.array([state]),
                    idx=self.sim_obj.model.c_idx["r"])
                output = np.array([final_size])
                return output
            elif self.epi_model == "seirSV_model":
                final_vaccinated = self.sim_obj.model.aggregate_by_age(
                    solution=np.array([state]),
                    idx=self.sim_obj.model.c_idx["v"])
                output = np.array([final_vaccinated])
                return output





