class StateCalculator:
    def __init__(self, sim_obj, epi_model):
        self.sim_obj = sim_obj
        self.epi_model = epi_model

    def calculate_infecteds(self, sol):
        if self.epi_model == "seir":
            return self._calculate_infecteds_seir(sol=sol)
        elif self.epi_model == "rost":
            return self._calculate_infecteds_rost(sol=sol)
        elif self.epi_model == "moghadas":
            return self._calculate_infecteds_moghadas(sol=sol)
        elif self.epi_model == "chikina":
            return self._calculate_infecteds_chikina(sol=sol)
        else:
            raise ValueError("Invalid epi_model")

    def _calculate_infecteds_seir(self, sol):
        # Calculate the number of infected individuals in the SEIR model
        n_infecteds = (
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["e"]) +
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["i"])
        )
        return n_infecteds

    def _calculate_infecteds_rost(self, sol):
        # Calculate the number of infected individuals in the Rost model
        n_infecteds = (
            sol.sum(axis=1) -
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["s"]) -
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["r"]) -
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["d"]) -
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["c"]) -
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["hosp"]) -
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["icu"])
        )
        return n_infecteds

    def _calculate_infecteds_moghadas(self, sol):
        # Calculate the number of infected individuals in the Moghadas model
        n_infecteds = (
            sol.sum(axis=1) -
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["s"]) -
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["r"]) -
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["d"]) -
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["i"]) -
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["hosp"]) -
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["icu"])
        )
        return n_infecteds

    def _calculate_infecteds_chikina(self, sol):
        # Calculate the number of infected individuals in the Chikina model
        n_infecteds = (
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["i"]) +
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["cp"]) +
            self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["c"])
        )
        return n_infecteds[0]

    def calculate_epidemic_peaks(self, sol):
        if self.epi_model == "rost":
            infecteds_peak = (
                sol.sum(axis=1) -
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["s"]) -
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["r"]) -
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["d"]) -
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["c"]) -
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["hosp"]) -
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["icu"])
            ).max()
        elif self.epi_model == "chikina":
            infecteds_peak = (
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["i"]) +
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["cp"]) +
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["c"])
            ).max()
        elif self.epi_model == "moghadas":
            infecteds_peak = (
                sol.sum(axis=1) -
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["s"]) -
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["r"]) -
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["d"]) -
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["i"]) -
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["hosp"]) -
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["icu"])
            ).max()
        elif self.epi_model == "seir":
            # Calculate the infected peak in the SEIR model
            infecteds_peak = (
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["e"]) +
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["i"])
            ).max()
        else:
            raise Exception("Invalid model!")
        return infecteds_peak

    def calculate_hospital_peak(self, sol):
        if self.epi_model == "rost":
            hospital_peak_now = (
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["ih"]) +
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["ic"]) +
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["icr"])
            ).max()
        elif self.epi_model == "chikina":
            hospital_peak_now = (
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["cp"]) +
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["c"])
            ).max()
        elif self.epi_model == "moghadas":
            hospital_peak_now = (
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["i_h"]) +
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["q_h"]) +
                self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["h"])
            ).max()
        else:
            hospital_peak_now = 0
        return hospital_peak_now

    def calculate_icu(self, sol):
        if self.epi_model == "rost":
            icu_now = self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["ic"]
                                                          ).max()
        else:
            icu_now = self.sim_obj.model.aggregate_by_age(solution=sol, idx=self.sim_obj.model.c_idx["c"]
                                                          ).max()
        return icu_now

    def calculate_final_size_dead(self, sol):
        if self.epi_model in ["rost", "chikina", "moghadas"]:
            state = sol[-1].reshape((1, -1))
            final_size_dead = self.sim_obj.model.aggregate_by_age(solution=state, idx=self.sim_obj.model.c_idx["d"])
        else:
            final_size_dead = 0
        return final_size_dead



