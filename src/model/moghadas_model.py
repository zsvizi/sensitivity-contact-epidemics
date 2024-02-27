import numpy as np

from src.model.model_base import EpidemicModelBase


class MoghadasModelUsa(EpidemicModelBase):
    def __init__(self, model_data) -> None:
        self.model_data = model_data
        compartments = ["s", "e", "i_n",
                        "q_n", "i_h", "q_h", "a_n",
                        "a_q", "h", "c", "r", "d"]
        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        iv["i_n"][2] = 1
        keys = ["e", "q_n", "i_h", "q_h", "a_n", "a_q", "h", "c", "r", "d"]
        iv.update({key: iv[key] for key in keys})
        iv.update({"s": self.population - (iv["e"] + iv["i_n"] + iv["q_n"] +
                                           iv["i_h"] + iv["q_h"] + iv["a_n"] +
                                           iv["a_q"] + iv["h"] + iv["c"] +
                                           iv["r"] + iv["d"])})

    def get_model(self, xs: np.ndarray, _, ps: dict, cm: np.ndarray) -> np.ndarray:
        s, e, i_n, q_n, i_h, q_h, a_n, a_q, h, c, r, d = xs.reshape(-1, self.n_age)
        transmission = ps["beta"] * np.array(i_n / self.population).dot(cm) + \
                       np.array(i_h / self.population).dot(cm) + \
                       np.array(ps["k"] * a_n / self.population).dot(cm)

        transmission_2 = ps["beta"] * np.array(q_n / self.population).dot(self.model_data.contact_data["Home"]) + \
                         np.array(q_h / self.population).dot(self.model_data.contact_data["Home"]) + \
                         np.array(ps["k"] * a_q / self.population).dot(self.model_data.contact_data["Home"])

        model_eq_dict = {
            "s": -ps["susc"] * s * transmission - ps["susc"] * s * transmission_2,  # S'(t)
            "e": ps["susc"] * s * transmission + ps["susc"] * s * transmission_2 - ps["sigma"] * e,  # E'(t)
            "i_n": (1 - ps["theta"]) * (1 - ps["q"]) * (1 - ps["h"]) * ps["sigma"] * e - \
                   (1 - ps["f_i"]) * ps["gamma"] * i_n - ps["f_i"] * ps["tau_i"] * i_n,  # In'(t)
            "q_n": (1 - ps["theta"]) * ps["q"] * (1 - ps["h"]) * ps["sigma"] * e - ps["gamma"] * \
                   q_n + ps["f_i"] * ps["tau_i"] * i_n,  # Qn'(t)
            "i_h": (1 - ps["theta"]) * (1 - ps["q"]) * ps["h"] * ps["sigma"] * e - \
                   (1 - ps["f_i"]) * ps["delta"] * i_h - ps["f_i"] * ps["tau_i"] * i_h,  # Ih'(t)
            "q_h": (1 - ps["theta"]) * ps["q"] * ps["h"] * ps["sigma"] * e - \
                   ps["delta"] * q_h + ps["f_i"] * ps["tau_i"] * i_h,  # Qh'(t)
            "a_n": ps["theta"] * ps["sigma"] * e - (1 - ps["f_a"]) * ps["delta"] * a_n - \
                   ps["f_a"] * ps["tau_a"] * a_n,  # An'(t)
            "a_q": ps["f_a"] * ps["tau_a"] * a_n - ps["gamma"] * a_q,  # Aq'(t)
            "h": (1 - ps["c"]) * (1 - ps["f_i"]) * ps["delta"] * i_h + (1 - ps["c"]) * \
                 ps["delta"] * q_h - (ps["m_h"] * ps["mu_h"] + (1 - ps["m_h"]) * ps["psi_h"]) * h,  # H'(t)
            "c": ps["c"] * (1 - ps["f_i"]) * ps["delta"] * i_h + ps["c"] * ps["delta"] * \
                 q_h - (ps["m_c"] * ps["mu_c"] + (1 - ps["m_c"]) * ps["psi_c"]) * c,  # C'(t)

            # added these two equations
            "r": ps["gamma"] * a_q + ps["m_h"] * h + ps["m_c"] * c,  # R'(t)
            "d": ps["m_h"] * ps["mu_h"] * h + ps["m_c"] * ps["mu_c"] * c  # D' (t)

        }

        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_hospitalized(self, solution: np.ndarray) -> np.ndarray:
        idx = self.c_idx["i_h"]
        idx_2 = self.c_idx["q_h"]
        idx_3 = self.c_idx["h"]
        return self.aggregate_by_age(solution, idx) + \
               self.aggregate_by_age(solution, idx_2) + self.aggregate_by_age(solution, idx_3)

    def get_icu_cases(self, solution: np.ndarray) -> np.ndarray:
        idx = self.c_idx["c"]
        return self.aggregate_by_age(solution, idx)
