import numpy as np
import src
from src.seirsv.dataloader import seirSVDataLoader


class Calculator:
    def __init__(self, data: seirSVDataLoader, sim_obj: src.SimulationNPI):
        self.data = data
        self.sim_obj = sim_obj

    def averted_infections_calc(self, params, cm: np.ndarray):
        """
                Calculates the sum of incident cases from time T at the start of day d
                to the start of the following day at time T + 1, then get
                averted number of infections in age group i,
                :return: averted number of infections / 100,000
                """
        t = np.arange(0, 11 * 1000, 1)    # run simulations from 2009 to 2020
        sol = self.sim_obj.model.get_solution(
            init_values=self.sim_obj.model.get_initial_values,
            t=t,
            parameters=params,
            cm=cm)
        state = sol[-1]
        infected = self.sim_obj.model.aggregate_by_age(
            solution=np.array([state]),
            idx=self.sim_obj.model.c_idx["i"])

        output = np.array([infected])

        # calculate the age dependent force of infection
        return np.array([self.sim_obj.model.no_seasonality_incident_cases])





