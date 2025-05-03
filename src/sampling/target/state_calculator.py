class StateCalculator:
    def __init__(self, sim_obj):
        self.sim_obj = sim_obj
        self.model = sim_obj.model

    def calculate_infecteds(self, sol):
        return self.model.get_infected(sol)

    def calculate_epidemic_peaks(self, sol):
        return self.model.get_epidemic_peak(sol)

    def calculate_hospital_peak(self, sol):
        return self.model.get_hospital_peak(sol)

    def calculate_icu(self, sol):
        return self.model.get_icu_cases(sol)

    def calculate_final_size_dead(self, sol):
        return self.model.get_final_size_dead(sol)
