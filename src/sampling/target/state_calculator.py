class StateCalculator:
    def __init__(self, sim_obj):
        self.sim_obj = sim_obj
        self.epi_model = sim_obj.epi_model

    def _infected_by_exclusion(self, sol, exclude_keys):
        total = sol.sum(axis=1)
        excluded = sum(
            self.sim_obj.model.aggregate_by_age(solution=sol,
                                                idx=self.sim_obj.model.c_idx[k])
            for k in exclude_keys
        )
        return total - excluded

    def _infected_by_inclusion(self, sol, include_keys):
        return sum(
            self.sim_obj.model.aggregate_by_age(solution=sol,
                                                idx=self.sim_obj.model.c_idx[k])
            for k in include_keys
        )

    def calculate_infecteds(self, sol):
        if self.epi_model in ["seir", "validation"]:
            return self._infected_by_inclusion(sol, ["e", "i"])
        elif self.epi_model in ["rost_maszk", "rost_prem"]:
            return self._infected_by_exclusion(sol, ["s", "r", "d", "c",
                                                     "hosp", "icu"])
        elif self.epi_model == "moghadas":
            return self._infected_by_exclusion(sol, ["s", "r", "d", "i",
                                                     "hosp", "icu"])
        elif self.epi_model == "chikina":
            return self._infected_by_inclusion(sol, ["i", "cp", "c"])[0]
        else:
            raise ValueError("Invalid epi_model")

    def calculate_epidemic_peaks(self, sol):
        if self.epi_model in ["rost_maszk", "rost_prem"]:
            return self._infected_by_exclusion(sol, ["s", "r", "d", "c",
                                                     "hosp", "icu"]).max()
        elif self.epi_model == "moghadas":
            return self._infected_by_exclusion(sol, ["s", "r", "d", "i",
                                                     "hosp", "icu"]).max()
        elif self.epi_model in ["seir", "validation"]:
            return self._infected_by_inclusion(sol, ["e", "i"]).max()
        elif self.epi_model == "chikina":
            return self._infected_by_inclusion(sol, ["i", "cp", "c"]).max()
        else:
            raise ValueError("Invalid epi_model")

    def calculate_hospital_peak(self, sol):
        if self.epi_model in ["rost_prem", "rost_maszk"]:
            hospital_peak_now = (
                self.sim_obj.model.aggregate_by_age(solution=sol,
                                                    idx=self.sim_obj.model.c_idx["ih"]) +
                self.sim_obj.model.aggregate_by_age(solution=sol,
                                                    idx=self.sim_obj.model.c_idx["ic"]) +
                self.sim_obj.model.aggregate_by_age(solution=sol,
                                                    idx=self.sim_obj.model.c_idx["icr"])
            ).max()
        elif self.epi_model == "chikina":
            hospital_peak_now = (
                self.sim_obj.model.aggregate_by_age(solution=sol,
                                                    idx=self.sim_obj.model.c_idx["cp"]) +
                self.sim_obj.model.aggregate_by_age(solution=sol,
                                                    idx=self.sim_obj.model.c_idx["c"])
            ).max()
        elif self.epi_model == "moghadas":
            hospital_peak_now = (
                self.sim_obj.model.aggregate_by_age(solution=sol,
                                                    idx=self.sim_obj.model.c_idx["i_h"]) +
                self.sim_obj.model.aggregate_by_age(solution=sol,
                                                    idx=self.sim_obj.model.c_idx["q_h"]) +
                self.sim_obj.model.aggregate_by_age(solution=sol,
                                                    idx=self.sim_obj.model.c_idx["h"])
            ).max()
        else:
            hospital_peak_now = 0
        return hospital_peak_now

    def calculate_icu(self, sol):
        if self.epi_model in ["rost_prem", "rost_maszk"]:
            icu_now = self.sim_obj.model.aggregate_by_age(
                solution=sol, idx=self.sim_obj.model.c_idx["ic"]
            ).max()
        elif self.epi_model in ["chikina", "moghadas"]:
            icu_now = self.sim_obj.model.aggregate_by_age(
                solution=sol, idx=self.sim_obj.model.c_idx["c"]
            ).max()
        else:
            icu_now = 0
        return icu_now

    def calculate_final_size_dead(self, sol):
        if self.epi_model in ["rost_prem", "rost_maszk", "chikina",
                              "moghadas", "validation"]:
            state = sol[-1].reshape((1, -1))
            return self.sim_obj.model.aggregate_by_age(
                solution=state, idx=self.sim_obj.model.c_idx["d"]
            )
        else:
            return 0
