import numpy as np

from src.model.model_base import EpidemicModelBase


class MoghadasModelUsa(EpidemicModelBase):
    """
    Age-structured epidemic model based on Moghadas et al. (USA variant).

    Compartments:
        s     – susceptible
        e     – exposed (latent)
        i_n   – infectious, not quarantined
        q_n   – quarantined infectious (non-hospitalized)
        i_h   – infectious requiring hospitalization (not yet hospitalized)
        q_h   – quarantined infectious requiring hospitalization
        a_n   – asymptomatic, not quarantined
        a_q   – asymptomatic, quarantined
        h     – hospitalized
        c     – critical/ICU
        r     – recovered
        d     – dead
        inf   – cumulative infections
        hosp  – cumulative hospital admissions
        icu   – cumulative ICU admissions
    """

    def __init__(self, model_data) -> None:
        self.model_data = model_data

        compartments = [
            "s", "e", "i_n",
            "q_n", "i_h", "q_h", "a_n",
            "a_q", "h", "c", "r", "d",
            "inf", "hosp", "icu"
        ]

        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        """
        Initializes compartment values before the ODE solver runs.

        - Seeds one infectious individual into age group 2 (i_n[2] = 1).
        - Recomputes cumulative compartments.
        - Updates susceptible population as:
              S = N − (all non-susceptible compartments)
        """

        # Seed 1 initial infectious case into age group index 2
        iv["i_n"][2] = 1

        # These keys are simply copied from iv to iv without modification.
        # This line is redundant, but harmless.
        keys = ["e", "q_n", "i_h", "q_h", "a_n", "a_q",
                "h", "c", "inf", "hosp", "icu"]
        iv.update({key: iv[key] for key in keys})

        # Susceptibles = population − all non-susceptible compartments
        iv.update({
            "s": self.population - (
                    iv["e"] + iv["i_n"] + iv["q_n"] +
                    iv["i_h"] + iv["q_h"] + iv["a_n"] +
                    iv["a_q"] + iv["h"] + iv["c"] +
                    iv["r"] + iv["d"]
            )
        })

    def get_model(self, xs: np.ndarray, t: int, ps: dict, cm: np.ndarray) -> np.ndarray:
        """
        Computes the ODE right-hand side.

        :param np.ndarray xs: State vector of size (n_age * n_compartments)
        :param int t: Current time
        :param dict ps: Dictionary of model parameters
        :param np.ndarray cm: Contact matrix (age x age)

        :return np.ndarray: Flattened array of age-structured derivatives.
        """

        # Unpack compartments in the exact declared order
        s, e, i_n, q_n, i_h, q_h, a_n, a_q, h, c, r, d, \
            inf, hosp, icu = xs.reshape(-1, self.n_age)

        # Community transmission
        transmission = (
                ps["beta"] * np.array(i_n / self.population).dot(cm) +
                np.array(i_h / self.population).dot(cm) +
                np.array(ps["k"] * a_n / self.population).dot(cm)
        )

        # Home transmission for quarantined individuals
        home_cm = self.model_data.contact_data["Home"]
        transmission_2 = (
                ps["beta"] * np.array(q_n / self.population).dot(home_cm) +
                np.array(q_h / self.population).dot(home_cm) +
                np.array(ps["k"] * a_q / self.population).dot(home_cm)
        )

        # Differential equations
        model_eq_dict = {
            # Susceptible
            "s": -ps["susc"] * s * transmission
                 - ps["susc"] * s * transmission_2,

            # Exposed
            "e": ps["susc"] * s * transmission
                 + ps["susc"] * s * transmission_2
                 - ps["sigma"] * e,

            # Infectious (not quarantined)
            "i_n": (
                    (1 - ps["theta"]) * (1 - ps["q"]) * (1 - ps["h"]) * ps["sigma"] * e
                    - (1 - ps["f_i"]) * ps["gamma"] * i_n
                    - ps["f_i"] * ps["tau_i"] * i_n
            ),

            # Quarantined infectious (non-hospital)
            "q_n": (
                    (1 - ps["theta"]) * ps["q"] * (1 - ps["h"]) * ps["sigma"] * e
                    - ps["gamma"] * q_n
                    + ps["f_i"] * ps["tau_i"] * i_n
            ),

            # Infectious requiring hospitalization
            "i_h": (
                    (1 - ps["theta"]) * (1 - ps["q"]) * ps["h"] * ps["sigma"] * e
                    - (1 - ps["f_i"]) * ps["delta"] * i_h
                    - ps["f_i"] * ps["tau_i"] * i_h
            ),

            # Quarantined infectious requiring hospitalization
            "q_h": (
                    (1 - ps["theta"]) * ps["q"] * ps["h"] * ps["sigma"] * e
                    - ps["delta"] * q_h
                    + ps["f_i"] * ps["tau_i"] * i_h
            ),

            # Asymptomatic (not quarantined)
            "a_n": (
                    ps["theta"] * ps["sigma"] * e
                    - (1 - ps["f_a"]) * ps["gamma"] * a_n
                    - ps["f_a"] * ps["tau_a"] * a_n
            ),

            # Asymptomatic quarantined
            "a_q": (
                    ps["f_a"] * ps["tau_a"] * a_n
                    - ps["gamma"] * a_q
            ),

            # Hospitalized
            # Outflows depend on mortality and recovery rates (mu_h, psi_h).
            "h": (
                    (1 - ps["c"]) * (1 - ps["f_i"]) * ps["delta"] * i_h
                    + (1 - ps["c"]) * ps["delta"] * q_h
                    - (ps["m_h"] * ps["mu_h"] + (1 - ps["m_h"]) * ps["psi_h"]) * h
            ),

            # ICU
            "c": (
                    ps["c"] * (1 - ps["f_i"]) * ps["delta"] * i_h
                    + ps["c"] * ps["delta"] * q_h
                    - (ps["m_c"] * ps["mu_c"] + (1 - ps["m_c"]) * ps["psi_c"]) * c
            ),

            # Recovered
            "r": (
                    ps["gamma"] * a_q
                    + ps["m_h"] * h
                    + ps["m_c"] * c
            ),

            # Deaths
            "d": (
                    ps["m_h"] * ps["mu_h"] * h
                    + ps["m_c"] * ps["mu_c"] * c
            ),

            # Cumulative incidence
            "inf": ps["sigma"] * e,

            # Cumulative hospital admissions
            "hosp": (
                    (1 - ps["theta"]) * (1 - ps["q"]) * ps["h"] * ps["sigma"] * e +
                    (1 - ps["theta"]) * ps["q"] * ps["h"] * ps["sigma"] * e +
                    ps["f_i"] * ps["tau_i"] * i_h +
                    (1 - ps["c"]) * (1 - ps["f_i"]) * ps["delta"] * i_h +
                    (1 - ps["c"]) * ps["delta"] * q_h
            ),

            # Cumulative ICU admissions
            "icu": (
                    ps["c"] * (1 - ps["f_i"]) * ps["delta"] * i_h +
                    ps["c"] * ps["delta"] * q_h
            ),
        }

        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_infected(self, solution: np.ndarray) -> np.ndarray:
        """
        Computes the number of currently infected individuals.

        Removes:
            - susceptible
            - recovered
            - dead
            - cumulative infection counts (inf)
            - cumulative hospitalizations (hosp)
            - cumulative ICU counts (icu)
        :return np.ndarray: The number of currently infected individuals.
        """

        total = solution.sum(axis=1)
        s = self.aggregate_by_age(solution, self.c_idx["s"])
        r = self.aggregate_by_age(solution, self.c_idx["r"])
        d = self.aggregate_by_age(solution, self.c_idx["d"])
        i = self.aggregate_by_age(solution, self.c_idx["inf"])
        hosp = self.aggregate_by_age(solution, self.c_idx["hosp"])
        icu = self.aggregate_by_age(solution, self.c_idx["icu"])

        return total - s - r - d - i - hosp - icu

    def get_epidemic_peak(self, solution: np.ndarray) -> float:
        """
        :return float: the maximum number of simultaneously infected individuals.
        """
        return self.get_infected(solution).max()

    def get_hospital_peak(self, solution: np.ndarray) -> float:
        """
        :return float: the peak number of individuals in hospital-related compartments: i_h + q_h + h
        """
        return (
                self.aggregate_by_age(solution, self.c_idx["i_h"]) +
                self.aggregate_by_age(solution, self.c_idx["q_h"]) +
                self.aggregate_by_age(solution, self.c_idx["h"])
        ).max()

    def get_icu_cases(self, solution: np.ndarray) -> float:
        """
        :return float: peak ICU occupancy.
        """
        return self.aggregate_by_age(solution, self.c_idx["c"]).max()

    def get_final_size_dead(self, solution: np.ndarray) -> float:
        """
        :return float: the total number of deaths at the end of the simulation.
        """
        final_state = solution[-1].reshape((1, -1))
        return self.aggregate_by_age(final_state, self.c_idx["d"])
