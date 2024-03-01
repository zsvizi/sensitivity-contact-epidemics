import numpy as np
from src.simulation_npi import SimulationNPI
from src.sampling.target_calculator import TargetCalculator


class FinalSizeTargetCalculator(TargetCalculator):
    def __init__(self, sim_obj: SimulationNPI, epi_model: str = "rost"):
        self.epi_model = epi_model
        super().__init__(sim_obj=sim_obj)

    def calculate_infecteds(self, state):
        if self.epi_model == "seir":
            return self._calculate_infecteds_seir(state)
        elif self.epi_model == "rost":
            return self._calculate_infecteds_rost(state)
        elif self.epi_model == "moghadas":
            return self._calculate_infecteds_moghadas(state)
        elif self.epi_model == "chikina":
            return self._calculate_infecteds_chikina(state)
        else:
            raise ValueError("Invalid epi_model")

    def _calculate_infecteds_seir(self, state):
        # Calculate the number of infected individuals in the SEIR model
        n_infecteds = (self.sim_obj.model.aggregate_by_age(
            solution=np.array([state]),
            idx=self.sim_obj.model.c_idx["e"]) +
                       self.sim_obj.model.aggregate_by_age(
            solution=np.array([state]), idx=self.sim_obj.model.c_idx["i"])
        )
        if n_infecteds < 1:
            return None  # Return None if infected count is less than 1
        else:
            return n_infecteds

    def _calculate_infecteds_rost(self, state):
        # Calculate the number of infected individuals in the Rost model
        state = np.array([state])
        n_infecteds = (state.sum() -
                       self.sim_obj.model.aggregate_by_age(
                           solution=state, idx=self.sim_obj.model.c_idx["s"]) -
                       self.sim_obj.model.aggregate_by_age(
                           solution=state, idx=self.sim_obj.model.c_idx["r"]) -
                       self.sim_obj.model.aggregate_by_age(
                           solution=state, idx=self.sim_obj.model.c_idx["d"]) -
                       self.sim_obj.model.aggregate_by_age(
                           solution=state, idx=self.sim_obj.model.c_idx["c"]))
        if n_infecteds < 1:
            return None  # Return None if infected count is less than 1
        else:
            return n_infecteds

    def _calculate_infecteds_moghadas(self, state):
        # Calculate the number of infected individuals in the Moghadas model
        state = np.array([state])
        n_infecteds = (state.sum() -
                       self.sim_obj.model.aggregate_by_age(
                           solution=state, idx=self.sim_obj.model.c_idx["s"]) -
                       self.sim_obj.model.aggregate_by_age(
                           solution=state, idx=self.sim_obj.model.c_idx["r"]) -
                       self.sim_obj.model.aggregate_by_age(
                           solution=state, idx=self.sim_obj.model.c_idx["d"]))
        if n_infecteds < 1:
            return None  # Return None if infected count is less than 1
        else:
            return n_infecteds

    def _calculate_infecteds_chikina(self, state):
        # Calculate the number of infected individuals in the Chikina model
        n_infecteds = self.sim_obj.model.aggregate_by_age(
            solution=np.array([state]),
            idx=self.sim_obj.model.c_idx["i"]) + \
                      self.sim_obj.model.aggregate_by_age(
                          solution=np.array([state]),
                          idx=self.sim_obj.model.c_idx["cp"]) + \
                      self.sim_obj.model.aggregate_by_age(
                          solution=np.array([state]),
                          idx=self.sim_obj.model.c_idx["c"])
        if n_infecteds < 1:
            return None  # Return None if infected count is less than 1
        else:
            return n_infecteds

    def calculate_epidemic_peaks(self, state):
        state = np.array([state])
        if self.epi_model == "rost":
            infecteds_peak = (state.sum() -
                           self.sim_obj.model.aggregate_by_age(
                               solution=state, idx=self.sim_obj.model.c_idx["s"]) -
                           self.sim_obj.model.aggregate_by_age(
                               solution=state, idx=self.sim_obj.model.c_idx["r"]) -
                           self.sim_obj.model.aggregate_by_age(
                               solution=state, idx=self.sim_obj.model.c_idx["d"]) -
                           self.sim_obj.model.aggregate_by_age(
                               solution=state, idx=self.sim_obj.model.c_idx["c"])
                              ).max()
        elif self.epi_model == "chikina":
            infecteds_peak = (self.sim_obj.model.aggregate_by_age(
                solution=state, idx=self.sim_obj.model.c_idx["i"]) +
                              self.sim_obj.model.aggregate_by_age(
                              solution=state,  idx=self.sim_obj.model.c_idx["cp"]) +
                              self.sim_obj.model.aggregate_by_age(
                              solution=state,
                              idx=self.sim_obj.model.c_idx["c"])).max()
        elif self.epi_model == "moghadas":
            infecteds_peak = (state.sum() -
                           self.sim_obj.model.aggregate_by_age(
                               solution=state, idx=self.sim_obj.model.c_idx["s"]) -
                           self.sim_obj.model.aggregate_by_age(
                               solution=state, idx=self.sim_obj.model.c_idx["r"]) -
                           self.sim_obj.model.aggregate_by_age(
                               solution=state, idx=self.sim_obj.model.c_idx["d"])
            ).max()
        elif self.epi_model == "seir":
            # Calculate the infected peak in the SEIR model
            infecteds_peak = (self.sim_obj.model.aggregate_by_age(
                solution=state,
                idx=self.sim_obj.model.c_idx["e"]) +
                              self.sim_obj.model.aggregate_by_age(
                                  solution=state, idx=self.sim_obj.model.c_idx["i"])
                              ).max()
        return infecteds_peak

    def calculate_hospital_peak(self, sol):
        if self.epi_model == "rost":
            hospital_peak_now = (
                    self.sim_obj.model.aggregate_by_age(
                        solution=sol, idx=self.sim_obj.model.c_idx["ih"]) +
                    self.sim_obj.model.aggregate_by_age(
                        solution=sol, idx=self.sim_obj.model.c_idx["ic"]) +
                    self.sim_obj.model.aggregate_by_age(
                        solution=sol, idx=self.sim_obj.model.c_idx["icr"])
            ).max()
        elif self.epi_model == "chikina":
            hospital_peak_now = (
                    self.sim_obj.model.aggregate_by_age(
                        solution=sol, idx=self.sim_obj.model.c_idx["cp"]) +
                    self.sim_obj.model.aggregate_by_age(solution=sol,
                                                        idx=self.sim_obj.model.c_idx["c"])
            ).max()
        elif self.epi_model == "moghadas":
            hospital_peak_now = (self.sim_obj.model.aggregate_by_age(
                    solution=sol, idx=self.sim_obj.model.c_idx["i_h"]) +
                                 self.sim_obj.model.aggregate_by_age(
                    solution=sol, idx=self.sim_obj.model.c_idx["q_h"]) +
                                 self.sim_obj.model.aggregate_by_age(
                    solution=sol, idx=self.sim_obj.model.c_idx["h"])
                                 ).max()
        return hospital_peak_now

    def calculate_icu(self, state):
        if self.epi_model == "rost":
            icu_now = self.sim_obj.model.aggregate_by_age(
                solution=state,
                idx=self.sim_obj.model.c_idx["ic"])
        else:
            icu_now = self.sim_obj.model.aggregate_by_age(
                solution=state,
                idx=self.sim_obj.model.c_idx["c"])
        return icu_now

    def calculate_final_size_dead(self, sol):
        state = np.array([sol[-1]])
        final_size_dead = self.sim_obj.model.aggregate_by_age(
            solution=state,
            idx=self.sim_obj.model.c_idx["d"])
        return final_size_dead

    def solve_model(self, t, cm):
        return self.sim_obj.model.get_solution(
            init_values=self.sim_obj.model.get_initial_values(),
            t=t,
            parameters=self.sim_obj.params,
            cm=cm)

    def get_output(self, cm: np.ndarray, include_infecteds=False,
                   include_infecteds_peak=False,
                   include_hospital_peak=False, include_icu=False,
                   include_final_size_dead=True):
        """
         Get output metrics for different targets.
         Args:
        cm (np.ndarray): The contact matrix used in the simulation.
        include_infecteds (bool, optional): Whether to include the number of infected individuals.
        Default is True.
        include_infecteds_peak (bool, optional): Whether to include the peak number of infected individuals.
            Default is False.
        include_hospital_peak (bool, optional): Whether to include the peak number of hospitalized individuals.
            Default is False.
        include_icu (bool, optional): Whether to include the number of individuals in intensive care units (ICU).
            Default is False.
        include_final_size_dead (bool, optional): Whether to include the final size of deceased individuals.
            Default is False.
        Returns:
        Tuple[Optional[np.ndarray]: A tuple containing the selected output metrics. Each element in the tuple
              corresponds to one of the selected metrics, or None if the metric was not selected.
    """
        t_interval = 100
        t = np.arange(0, t_interval, 0.5)
        t_interval_complete = 0

        sol = self.solve_model(t, cm)
        state = sol[-1]
        hospital_peak = (
                self.sim_obj.model.aggregate_by_age(
                    solution=sol, idx=self.sim_obj.model.c_idx["ih"]) +
                self.sim_obj.model.aggregate_by_age(
                    solution=sol, idx=self.sim_obj.model.c_idx["ic"]) +
                self.sim_obj.model.aggregate_by_age(
                    solution=sol, idx=self.sim_obj.model.c_idx["icr"])
        ).max()

        while True:
            hospital_peak_now = self.calculate_hospital_peak(sol)
            # check whether it is higher than it was in the previous turn
            if hospital_peak_now > hospital_peak:
                hospital_peak = hospital_peak_now
                hospital_peak_now = np.array([hospital_peak_now])

            infecteds = self.calculate_infecteds(state)
            if infecteds < 1:
                break

            # since the number of infecteds is above 1, we solve the ODE again
            # from 0 to t_interval using the current state
            sol = self.solve_model(t, cm)
            t_interval_complete += t_interval
            state = sol[-1]

            infecteds = self.calculate_infecteds(state)
            infecteds_peak = self.calculate_epidemic_peaks(state=state)
            hospital_peak = hospital_peak_now
            final_size_dead = self.calculate_final_size_dead(sol=sol)
            icu = self.calculate_icu(np.array([state]))

            # return the targets
            if include_final_size_dead:
                return np.array([final_size_dead])
            elif include_icu:
                return np.array([icu])
            elif include_hospital_peak:
                return np.array([hospital_peak_now])
            elif include_infecteds_peak:
                return np.array([infecteds_peak])
            elif include_infecteds:
                return np.array([infecteds])



