import numpy as np
from src.simulation.simulation_npi import SimulationNPI
from src.sampling.target.state_calculator import StateCalculator
from src.sampling.target.target_calculator import TargetCalculator


class ODETargetCalculator(TargetCalculator):
    """
    Target calculator for running ODE-based epidemiological simulations.

    This class integrates the model over time using the given contact matrix,
    until the epidemic naturally decays (infected individuals drop below 1).
    It then computes a configurable set of epidemic indicators such as:
    - Final death size
    - ICU peak
    - Hospitalization peak
    - Infection peak
    - Total number of infected individuals
    """

    def __init__(self, sim_obj: SimulationNPI):
        """
        Initialize the ODE target calculator.

        :param SimulationNPI sim_obj: Simulation object containing the
                                      epidemiological model, parameters,
                                      and configuration.
        """
        super().__init__(sim_obj=sim_obj)
        self.config = sim_obj.config
        self.state_calc = StateCalculator(sim_obj=sim_obj)

    def get_output(self, cm: np.ndarray) -> np.ndarray:
        """
        Integrates the model ODEs over time and compute epidemic targets.

        The simulation runs in time segments until the number of infected individuals falls below 1,
        indicating the end of the epidemic. After integration, this method computes and returns selected
        epidemiological metrics according to the configuration.

        :param np.ndarray cm: Contact matrix used in the simulation run.
        :return np.ndarray: Array of computed epidemiological target values.
        """
        # Define base simulation time segment
        t_interval = 250
        t = np.arange(0, t_interval, 0.5)
        t_interval_complete = 0  # TODO: ez mire kell? nem jó semmire jelenleg

        # Run initial ODE integration
        sol = self.sim_obj.model.get_solution(
            init_values=self.sim_obj.model.get_initial_values(),
            t=t,
            parameters=self.sim_obj.params,
            cm=cm
        )

        complete_sol = sol.copy()  # Store the evolving epidemic trajectory
        state = sol[-1]  # Last state of the current time segment

        # Continue simulation while infection count remains significant
        while True:
            infecteds = self.state_calc.calculate_infecteds(sol=np.array([state]))
            if infecteds < 1:
                # Epidemic ended (less than one infected individual)
                break

            # Continue simulation from the last known state for another interval
            sol = self.sim_obj.model.get_solution(
                init_values=state,
                t=t,
                parameters=self.sim_obj.params,
                cm=cm
            )

            t_interval_complete += t_interval
            state = sol[-1]

            # Append new results (excluding repeated initial state)
            complete_sol = np.append(complete_sol, sol[1:, :], axis=0)

        # Collect output metrics based on the simulation configuration
        output = []

        if self.config["include_final_death_size"]:
            final_size_dead = self.state_calc.calculate_final_size_dead(sol=complete_sol)
            output.append(final_size_dead[0])

        if self.config["include_icu_peak"]:
            icu_peak = self.state_calc.calculate_icu(sol=complete_sol)
            output.append(icu_peak)

        if self.config["include_hospital_peak"]:
            hospital_peak = self.state_calc.calculate_hospital_peak(sol=complete_sol)
            output.append(hospital_peak)

        if self.config["include_infecteds_peak"]:
            infecteds_peak = self.state_calc.calculate_epidemic_peaks(sol=complete_sol)
            output.append(infecteds_peak)

        if self.config["include_infecteds"]:
            total_infecteds = self.state_calc.calculate_infecteds(sol=complete_sol)
            output.append(total_infecteds)

        # Convert the list of metrics into a NumPy array
        return np.array(output)
