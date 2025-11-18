import numpy as np

from src.model.model_base import EpidemicModelBase


class RostModelHungary(EpidemicModelBase):
    """
    Age-structured SEIR-type epidemic model used for the Hungarian scenario.

    Compartment structure:
        s    – susceptible
        l1   – latent stage 1
        l2   – latent stage 2
        ip   – presymptomatic infectious
        ia1, ia2, ia3 – asymptomatic infectious stages
        is1, is2, is3 – symptomatic infectious stages
        ih   – hospitalized
        ic   – ICU
        icr  – recovering from ICU (step-down)
        r    – recovered
        d    – deceased
        inf  – cumulative infections (incidence)
        hosp – cumulative hospital admissions
        icu  – cumulative ICU admissions
    """

    def __init__(self, model_data) -> None:
        compartments = [
            "s", "l1", "l2",
            "ip", "ia1", "ia2", "ia3",
            "is1", "is2", "is3",
            "ih", "ic", "icr",
            "r", "d", "inf", "hosp", "icu"
        ]
        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        """
        Updates initial values before solving the ODE.
        The model seeds a single latent case into age group 2.

        Also updates:
            - `inf`: initial cumulative infections
            - `s`: susceptible population = N − (infected + latent)
        """
        # Seed one case in latent stage L1, age group index 2
        iv["l1"][2] = 1

        # Compute initial cumulative infections
        iv.update({
            "inf": iv["ip"] + iv["ia1"] + iv["ia2"] + iv["ia3"] +
                   iv["is1"] + iv["is2"] + iv["is3"] +
                   iv["r"] + iv["d"]
        })

        # Susceptibles = population − (currently infected + latent)
        iv.update({
            "s": self.population - (iv["inf"] + iv["l1"] + iv["l2"])
        })

    def get_model(self, xs: np.ndarray, t: int, ps: dict, cm: np.ndarray) -> np.ndarray:
        """
        Computes the right-hand side of the ODE system.

        :param np.ndarray xs: State vector of size (n_age * n_compartments).
        :param int t: Current time
        :param np.ndarray ps: Model parameters.
        :param np.ndarray cm: Contact matrix (age * age).

        :return np.ndarray: The flattened array of derivatives for all age groups.
        """

        # Unpack compartments (order must match self.compartments)
        s, l1, l2, ip, ia1, ia2, ia3, is1, is2, is3, ih, ic, icr, \
            r, d, inf, hosp, icu = xs.reshape(-1, self.n_age)

        # Force of infection (for each receiving age group)
        transmission = ps["beta"] * np.array(
            ip + ps["inf_a"] * (ia1 + ia2 + ia3) + (is1 + is2 + is3)
        ).dot(cm)

        actual_population = self.population

        # System of differential equations
        model_eq_dict = {
            # Susceptible
            "s": -ps["susc"] * (s / actual_population) * transmission,

            # Latent stages
            "l1": ps["susc"] * (s / actual_population) * transmission - 2 * ps["alpha_l"] * l1,
            "l2": 2 * ps["alpha_l"] * l1 - 2 * ps["alpha_l"] * l2,

            # Presymptomatic
            "ip": 2 * ps["alpha_l"] * l2 - ps["alpha_p"] * ip,

            # Asymptomatic chain (3 stages)
            "ia1": ps["p"] * ps["alpha_p"] * ip - 3 * ps["gamma_a"] * ia1,
            "ia2": 3 * ps["gamma_a"] * ia1 - 3 * ps["gamma_a"] * ia2,
            "ia3": 3 * ps["gamma_a"] * ia2 - 3 * ps["gamma_a"] * ia3,

            # Symptomatic chain (3 stages)
            "is1": (1 - ps["p"]) * ps["alpha_p"] * ip - 3 * ps["gamma_s"] * is1,
            "is2": 3 * ps["gamma_s"] * is1 - 3 * ps["gamma_s"] * is2,
            "is3": 3 * ps["gamma_s"] * is2 - 3 * ps["gamma_s"] * is3,

            # Hospitalization
            # Inflow is fraction h * (1 - xi), outflow at gamma_h
            "ih": ps["h"] * (1 - ps["xi"]) * 3 * ps["gamma_s"] * is3 - ps["gamma_h"] * ih,

            # ICU
            # Inflow is fraction h * xi, outflow at gamma_c
            "ic": ps["h"] * ps["xi"] * 3 * ps["gamma_s"] * is3 - ps["gamma_c"] * ic,

            # ICU recovery (step-down)
            "icr": (1 - ps["mu"]) * ps["gamma_c"] * ic - ps["gamma_cr"] * icr,

            # Recovered
            "r": (
                    3 * ps["gamma_a"] * ia3 +
                    (1 - ps["h"]) * 3 * ps["gamma_s"] * is3 +  # symptomatic directly recovering
                    ps["gamma_h"] * ih +  # hospital recoveries
                    ps["gamma_cr"] * icr  # ICU step-down recoveries
            ),

            # Deaths
            "d": ps["mu"] * ps["gamma_c"] * ic,

            # Cumulative infections (incidence)
            "inf": 2 * ps["alpha_l"] * l2,

            # Cumulative hospitalization
            "hosp": (
                    ps["h"] * (1 - ps["xi"]) * 3 * ps["gamma_s"] * is3 +
                    ps["h"] * ps["xi"] * 3 * ps["gamma_s"] * is3
            ),

            # Cumulative ICU admissions
            "icu": ps["h"] * ps["xi"] * 3 * ps["gamma_s"] * is3
        }

        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_infected(self, solution: np.ndarray) -> np.ndarray:
        """
        :return np.ndarray: total infected population (excluding S, R, D, cumulative counts).
        """
        total = solution.sum(axis=1)
        s = self.aggregate_by_age(solution, self.c_idx["s"])
        r = self.aggregate_by_age(solution, self.c_idx["r"])
        d = self.aggregate_by_age(solution, self.c_idx["d"])
        c = self.aggregate_by_age(solution, self.c_idx["inf"])
        hosp = self.aggregate_by_age(solution, self.c_idx["hosp"])
        icu = self.aggregate_by_age(solution, self.c_idx["icu"])

        # Infected = everything except these removed/cumulative compartments
        return total - s - r - d - c - hosp - icu

    def get_epidemic_peak(self, solution: np.ndarray) -> float:
        """
        :return float: the maximum number of simultaneously infected individuals.
        """
        return self.get_infected(solution).max()

    def get_hospital_peak(self, solution: np.ndarray) -> float:
        """
        :return float: Peak value of total hospitalized (Ih + Ic + Icr).
        """
        ih = self.aggregate_by_age(solution, self.c_idx["ih"])
        ic = self.aggregate_by_age(solution, self.c_idx["ic"])
        icr = self.aggregate_by_age(solution, self.c_idx["icr"])
        return (ih + ic + icr).max()

    def get_icu_cases(self, solution: np.ndarray) -> float:
        """
        :return float: the peak ICU occupancy.
        """
        idx = self.c_idx["ic"]
        return self.aggregate_by_age(solution, idx).max()

    def get_final_size_dead(self, solution: np.ndarray) -> float:
        """
        :return float: the final number of deaths at the end of the simulation.
        """
        state = solution[-1].reshape((1, -1))
        return self.aggregate_by_age(state, self.c_idx["d"])
