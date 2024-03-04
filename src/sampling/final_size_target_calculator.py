import numpy as np
from src.simulation_npi import SimulationNPI
from src.sampling.state_calculator import StateCalculator
from src.sampling.target_calculator import TargetCalculator


class FinalSizeTargetCalculator(TargetCalculator):
    def __init__(self, sim_obj: SimulationNPI, epi_model: str = "rost"):
        super().__init__(sim_obj=sim_obj)
        self.config = {
            "include_infecteds": False,
            "include_infecteds_peak": False,
            "include_hospital_peak": False,
            "include_icu": False,
            "include_final_size_dead": True
        }
        self.state_calc = StateCalculator(sim_obj=sim_obj, epi_model=epi_model)

    def get_output(self, cm: np.ndarray):
        t_interval = 100
        t = np.arange(0, t_interval, 0.5)
        t_interval_complete = 0

        sol = self.sim_obj.model.get_solution(
            init_values=self.sim_obj.model.get_initial_values(),
            t=t,
            parameters=self.sim_obj.params,
            cm=cm
        )
        complete_sol = sol.copy()
        state = sol[-1]
        hospital_peak = (
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["ih"]) +
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["ic"]) +
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["icr"])
        ).max()

        while True:
            hospital_peak_now = self.state_calc.calculate_hospital_peak(sol)
            # check whether it is higher than it was in the previous turn
            if hospital_peak_now > hospital_peak:
                hospital_peak = hospital_peak_now
                hospital_peak_now = np.array([hospital_peak_now])

            infecteds = self.state_calc.calculate_infecteds(state)
            if infecteds < 1:
                break

            # since the number of infecteds is above 1, we solve the ODE again
            # from 0 to t_interval using the current state
            sol = self.sim_obj.model.get_solution(
                init_values=state,
                t=t,
                parameters=self.sim_obj.params,
                cm=cm)
            t_interval_complete += t_interval
            state = sol[-1]
            complete_sol = np.append(complete_sol, sol[1:])

        infecteds = self.state_calc.calculate_infecteds(sol=complete_sol)
        infecteds_peak = self.state_calc.calculate_epidemic_peaks(sol=complete_sol)
        final_size_dead = self.state_calc.calculate_final_size_dead(sol=complete_sol)
        icu = self.state_calc.calculate_icu(sol=np.array([complete_sol]))

        # return the targets
        if self.config["include_final_size_dead"]:
            return np.array([final_size_dead])
        elif self.config["include_icu"]:
            return np.array([icu])
        elif self.config["include_hospital_peak"]:
            return np.array([hospital_peak_now])
        elif self.config["include_infecteds_peak"]:
            return np.array([infecteds_peak])
        elif self.config["include_infecteds"]:
            return np.array([infecteds])
