import numpy as np
import math
from src.model.model_base import EpidemicModelBase


class SeirSVModel(EpidemicModelBase):
    def __init__(self, model_data, country: str = "UK", model_no_seasonality: str = "Yes",
                 model_seasonality: str = "Yes", vaccine_induced_halved: str = "Yes",
                 varying_vaccination_rate: str = "Yes", model_seed: str = "Yes",
                 same_vaccination: str = "Yes") -> None:

        self.same_vaccination = same_vaccination
        self.model_seed = model_seed
        self.varying_vaccination_rate = varying_vaccination_rate
        self.vaccine_induced_halved = vaccine_induced_halved
        self.no_seasonality = model_no_seasonality
        self.model_seasonality = model_seasonality

        self.vaccine_induced = []
        self.no_seasonality_incident_cases = []
        self.seasonality_incident_cases = []
        self.incident_cases_vac = []
        self.incident_cases_seed = []
        self.same_vac = []
        self.incident_current_prac = []
        self.country = country

        compartments = ["s", "e", "i", "r", "v"]

        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        iv["i"][3] = 1
        iv.update({"e": iv["e"], "r": iv["r"], "v": iv["v"]})

        iv.update({"s": self.population - (iv["e"] + iv["i"] + iv["r"] + iv["v"])})

    def calculate_incident_values(self, lambda_i):
        return np.sum(np.cumsum(lambda_i * self.population) / 100000)

    def calculate_force_of_infection(self, ps, i, cm):
        # Calculate force of infection based on age-dependent parameters
        return np.sum(ps["beta"] * np.array(i).dot(cm) / self.population)

    def get_model(self, xs: np.ndarray, t: int, ps, cm: np.ndarray) -> np.ndarray:
        """
        Implemented the transmission model in Appendix A, section 2 i.e. A.2
        The model uses parameters corresponding to Influenza A.
        :param xs: values of all the compartments as a numpy array
        :param t: number of days since the start of the simulation.
        :param ps: uk parameters stored as a dictionary
        :param cm: uk contact matrix
        :return: seirSV model as an array
        """
        s, e, i, r, v = xs.reshape(-1, self.n_age)
        lambda_i = self.calculate_force_of_infection(ps, i, cm)
        param = ps["mu_i"] + ps["psi_i"] + lambda_i
        ps.update({"param": param})
        seeding_start_time = 365  # seeding time
        # add seasonality
        z = 1 + ps["q"] * np.sin(2 * math.pi * (t - ps["t_offset"]) / 365)
        ps.update({"z": z})

        # No seasonality scenario
        if self.no_seasonality == "Yes":
            seirSV_eq_dict = self.get_seirSV_eq_dict(ps, xs, lambda_i)
            self.no_seasonality_incident_cases = self.calculate_incident_values(lambda_i)
        if self.model_seasonality == "Yes":
            lambda_i *= z
            seirSV_eq_dict = self.get_seirSV_eq_dict(ps, xs, lambda_i)
            self.seasonality_incident_cases = self.calculate_incident_values(lambda_i)
        # Vaccination scenario
        if self.varying_vaccination_rate == "Yes" and t >= 107:
            vary_vaccination_rate = ps["psi_i"] * (1 + ps["q"] * np.sin(2 * math.pi * (t + 107) / 365))
            ps.update({"vary_vaccination_rate": vary_vaccination_rate})
            lambda_i *= vary_vaccination_rate
            seirSV_eq_dict = self.get_seirSV_eq_dict(ps, xs, lambda_i)
            self.incident_cases_vac = self.calculate_incident_values(lambda_i)
        if self.model_seed == "Yes" and t >= seeding_start_time:
            # Seeding condition
            current_seeding = np.concatenate([np.full(self.n_age - 5, 10), np.zeros(5)])
            previous_seeding = np.concatenate([np.full(self.n_age - 5, 100), np.zeros(5)])
            current_seeding_infection = current_seeding if t < seeding_start_time else previous_seeding
            seeding_part = param + current_seeding_infection
            ps.update({"param": param,
                       "seeding": seeding_part})
            lambda_i = z * lambda_i
            lambda_i += current_seeding_infection / self.population
            seirSV_eq_dict = self.get_seirSV_eq_dict(ps, xs, lambda_i)
            self.incident_cases_seed = self.calculate_incident_values(lambda_i)
        if self.vaccine_induced_halved == "Yes":
            lambda_i *= z
            ps["omega_v"] = 0.833333
            seirSV_eq_dict = self.get_seirSV_eq_dict(ps, xs, lambda_i)
            self.vaccine_induced = self.calculate_incident_values(lambda_i)
        if self.same_vaccination == "Yes":
            lambda_i *= z
            p = np.full(15, 0.5)
            ps["psi_i"] = p
            seirSV_eq_dict = self.get_seirSV_eq_dict(ps, xs, lambda_i)
            self.same_vac = self.calculate_incident_values(lambda_i)
        return self.get_array_from_dict(comp_dict=seirSV_eq_dict)

    def get_seirSV_eq_dict(self, ps, xs: np.ndarray, lambda_i):
        s, e, i, r, v = xs.reshape(-1, self.n_age)
        seirSV_eq_dict = {"s": ps["omega_v"] * v + ps["omega_i"] *
                               r - s / self.population * [ps["mu_i"] + ps["psi_i"] +
                                                          lambda_i],  # S'(t)
                          "e": lambda_i * s / self.population - e * [ps["mu_i"] + ps["gamma"]],  # E'(t)
                          "i": ps["gamma"] * e - i * [ps["mu_i"] + ps["rho"]],  # I'(t)
                          "r": ps["rho"] * i - r * [ps["mu_i"] - ps["omega_i"]],  # R'(t)
                          "v": ps["psi_i"] * s / self.population - v * [ps["mu_i"] + ps["omega_v"]],  # V'(t)
                          }
        return seirSV_eq_dict

    def get_infected(self, solution) -> np.ndarray:
        idx = self.c_idx["i"]
        return self.aggregate_by_age(solution, idx)

    def get_recovered(self, solution) -> np.ndarray:
        idx = self.c_idx["r"]
        return self.aggregate_by_age(solution, idx)

    def get_vaccinated(self, solution) -> np.ndarray:
        idx = self.c_idx["v"]
        return self.aggregate_by_age(solution, idx)
