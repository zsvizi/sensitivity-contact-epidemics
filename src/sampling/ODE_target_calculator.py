import numpy as np
from src.simulation_npi import SimulationNPI
from src.sampling.state_calculator import StateCalculator
from src.sampling.target_calculator import TargetCalculator


class ODETargetCalculator(TargetCalculator):
    def __init__(self, sim_obj: SimulationNPI, config: dict, epi_model: str = "rost"):
        super().__init__(sim_obj=sim_obj)
        self.sim_output_values = {}  # track the target
        self.config = config
        self.state_calc = StateCalculator(sim_obj=sim_obj, epi_model=epi_model)

    def get_output(self, cm: np.ndarray):
        t_interval = 250
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

        while True:
            infecteds = self.state_calc.calculate_infecteds(sol=np.array([state]))
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
            complete_sol = np.append(complete_sol, sol[1:, :], axis=0)

        hospital_peak_now = self.state_calc.calculate_hospital_peak(sol=complete_sol)
        infecteds = self.state_calc.calculate_infecteds(sol=complete_sol)
        infecteds_peak = self.state_calc.calculate_epidemic_peaks(sol=complete_sol)
        final_size_dead = self.state_calc.calculate_final_size_dead(sol=complete_sol)
        icu = self.state_calc.calculate_icu(sol=complete_sol)

        output = []  # collecting the targets from epidemic size
        if self.config["include_final_death_size"]:
            self.sim_output_values["final_death_size"] = {
                "values":final_size_dead[0],
                "description": "Run simulations with final_death_size as target"
            }
            output.append(final_size_dead[0])

        if self.config["include_icu_peak"]:
            self.sim_output_values["icu_peak"] = {
                "values":icu,
                "description": "Run simulations with icu_peak as target"
            }
            output.append(icu)

        if self.config["include_hospital_peak"]:
            self.sim_output_values["hospital_peak"] = {
                "values": hospital_peak_now,
                "description": "Run simulations with hospital_peak as target"
            }
            output.append(hospital_peak_now)

        if self.config["include_infecteds_peak"]:
            self.sim_output_values["infecteds_peak"] = {
                "values": infecteds_peak,
                "description": "Run simulations with infecteds_peak as target"
            }
            output.append(infecteds_peak)

        if self.config["include_infecteds"]:
            output.append(infecteds)
        return np.array(output)