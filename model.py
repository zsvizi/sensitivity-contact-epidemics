import numpy as np
from scipy.integrate import odeint


class RostModelHungary:
    def __init__(self, model_data):
        self.population = model_data.age_data.flatten()

        self.compartments = ["s", "l1", "l2",
                             "ip", "ia1", "ia2", "ia3",
                             "is1", "is2", "is3",
                             "ih", "ic", "icr",
                             "r", "d", "c"]
        self.c_idx = {comp: idx for idx, comp in enumerate(self.compartments)}
        self.n_age = self.population.shape[0]

    def get_initial_values(self):
        iv = {
            "l1": np.zeros(self.n_age),
            "l2": np.zeros(self.n_age),

            "ip": np.zeros(self.n_age),

            "ia1": np.zeros(self.n_age),
            "ia2": np.zeros(self.n_age),
            "ia3": np.zeros(self.n_age),

            "is1": np.zeros(self.n_age),
            "is2": np.zeros(self.n_age),
            "is3": np.zeros(self.n_age),

            "ih": np.zeros(self.n_age),
            "ic": np.zeros(self.n_age),
            "icr": np.zeros(self.n_age),

            "d": np.zeros(self.n_age),
            "r": np.zeros(self.n_age)
        }
        iv["l1"][2] = 1
        iv.update({
            "c": iv["ip"] + iv["ia1"] + iv["ia2"] + iv["ia3"] + iv["is1"] + iv["is2"] + iv["is3"] + iv["r"] + iv["d"]
        })
        iv.update({
            "s": self.population - (iv["c"] + iv["l1"] + iv["l2"])
        })

        return np.array([iv[comp] for comp in self.compartments]).flatten()

    def get_solution(self, t, parameters, cm):
        initial_values = self.get_initial_values()
        return np.array(odeint(self.get_model, initial_values, t, args=(parameters, cm)))

    def get_model(self, xs, _, ps, cm):
        # the same order as in self.compartments!
        s, l1, l2, ip, ia1, ia2, ia3, is1, is2, is3, ih, ic, icr, r, d, c = xs.reshape(-1, 16)

        transmission = ps["beta"] * \
            np.array((ip + ps["inf_a"] * (ia1 + ia2 + ia3) + (is1 + is2 + is3))).dot(cm)
        actual_population = self.population

        # TODO: implement old RostModelHungary
        model_eq_dict = {
            "s": -ps["susc"] * (s / actual_population) * transmission,  # S'(t)
            "l1": ps["susc"] * (s / actual_population) * transmission - 2 * ps["alpha_l"] * l1,  # L1'(t)
            "l2": 2 * ps["alpha_l"] * l1 - 2 * ps["alpha_l"] * l2,  # L2'(t)

            "ip": (1 - ps["p"]) * 2 * ps["alpha_l"] * l2 - ps["alpha_p"] * ip,  # Ip'(t)

            "ia1": ps["p"] * 2 * ps["alpha_l"] * l2 - 3 * ps["gamma_a"] * ia1,  # Ia1'(t)
            "ia2": 3 * ps["gamma_a"] * ia1 - 3 * ps["gamma_a"] * ia2,  # Ia2'(t)
            "ia3": 3 * ps["gamma_a"] * ia2 - 3 * ps["gamma_a"] * ia3,  # Ia3'(t)

            "is1": ps["alpha_p"] * ip - 3 * ps["gamma_s"] * is1,  # Is1'(t)
            "is2": 3 * ps["gamma_s"] * is1 - 3 * ps["gamma_s"] * is2,  # Is2'(t)
            "is3": 3 * ps["gamma_s"] * is2 - 3 * ps["gamma_s"] * is3,  # Is3'(t)

            "ih": ps["h"] * (1 - ps["xi"]) * 3 * ps["gamma_s"] * is3
            - ps["gamma_h"] * ih,  # Ih'(t)
            "ic": ps["h"] * ps["xi"] * 3 * ps["gamma_s"] * is3
            - ps["gamma_c"] * ic,  # Ic'(t)
            "icr": (1 - ps["mu"]) * ps["gamma_c"] * ic - ps["gamma_cr"] * icr,  # Icr'(t)

            "r": 3 * ps["gamma_a"] * ia3 + (1 - ps["h"]) * 3 * ps["gamma_s"] * is3
            + ps["gamma_h"] * ih + ps["gamma_cr"] * icr,  # R'(t)
            "d": ps["mu"] * ps["gamma_c"] * ic,  # D'(t)

            "c": 2 * ps["alpha_l"] * l2  # C'(t)
        }

        model_eq = [model_eq_dict[comp] for comp in self.compartments]
        v = np.array(model_eq).flatten()
        return v

    @staticmethod
    def aggregate_by_age(solution, idx):
        return np.sum(solution[:, idx * 10:(idx + 1) * 10], axis=1)

    def get_cumulative(self, solution):
        idx = self.c_idx["c"]
        return self.aggregate_by_age(solution, idx)

    def get_deaths(self, solution):
        idx = self.c_idx["d"]
        return self.aggregate_by_age(solution, idx)

    def get_hospitalized(self, solution):
        idx = self.c_idx["ih"]
        idx_2 = self.c_idx["icr"]
        return self.aggregate_by_age(solution, idx) + self.aggregate_by_age(solution, idx_2)

    def get_ventilated(self, solution):
        idx = self.c_idx["ic"]
        return self.aggregate_by_age(solution, idx)
