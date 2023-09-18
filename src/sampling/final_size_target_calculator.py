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
        sol = None  # TODO
        # store the state at the last time point - this is referred as current state
        state = None  # TODO
        # introduce a variable for tracking how long the ODE was solved
        t_interval_complete = 0
        t_interval_complete += t_interval
        # let's say we want to store the peak size of hospitalized people during the simulation
        # aggregate age groups for ih, ic and icr from the previously calculated solution
        # then get maximal value of the time series
        hospital_peak = None  # TODO

        # Loop infinitely
        while True:
            # let's say we want to store the peak size of hospitalized people during the simulation
            # aggregate age groups for ih, ic and icr from the previously calculated solution
            # then get maximal value of the time series
            hospital_peak_now = None  # TODO
            # check whether it is higher than it was in the previous turn
            if None:  # TODO
                # if yes, then update the variable created outside this loop
                pass  # TODO

            # calculate the number of infected individuals at the current state
            # sum of aggregation along all age groups in
            # l1, l2, ip, ia1, ia2, ia3, is1, is2, is3, ih, ic, icr
            n_infecteds = None  # TODO
            # check whether the previously calculated number is less than 1
            if None:
                break

            # since the number of infecteds is above 1, we solve the ODE again
            # from 0 to t_interval using the current state
            sol = None  # TODO
            # then store the state at the last time point again and update the current state
            state = None  # TODO
            # update variable for tracking interval length on which the ODE is solved
            t_interval_complete += t_interval

        # for calculating final size value for compartment D
        # aggregate the current state (calculated in the last turn of the loop) for compartment "d"
        final_size_dead = None  # TODO
        # concatenate all output values to get the array of return
        output = None  # TODO
        return output
