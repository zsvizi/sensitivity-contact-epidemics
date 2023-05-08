import numpy as np
import scipy.stats as ss

from src.prcc import get_prcc_values, get_rectangular_matrix_from_upper_triu
from src.simulation_npi import SimulationNPI


class PRCCCalculator:
    def __init__(self, sim_obj: SimulationNPI,
                 number_of_samples: int):
        self.sim_obj = sim_obj
        self.n_ag = sim_obj.n_ag
        self.age_vector = sim_obj.age_vector
        self.params = sim_obj.params
        self.number_of_samples = number_of_samples
        self.upp_tri_size = int((self.n_ag + 1) * self.n_ag / 2)

        self.prcc_mtx = np.array([])
        self.p_icr = []
        self.prcc_matrix_school = []  # 16 * 16
        self.prcc_matrix_work = []
        self.prcc_matrix_other = []
        self.p_value = np.array([])
        self.agg_prcc = np.array([])
        self.agg_lock3 = np.array([])
        self.prcc_list = None
        self.p_value_mtx = None
        self.agg_pval = None

    def calculate_prcc_values(self, lhs_table: np.ndarray, sim_output: np.ndarray):
        sim_data = lhs_table[:, :(self.n_ag * (self.n_ag + 1)) // 2]
        sim_data = 1 - sim_data
        simulation = np.append(sim_data, sim_output[:, -self.n_ag - 1].reshape((-1, 1)), axis=1)
        prcc_list = get_prcc_values(simulation, number_of_samples=self.number_of_samples)
        prcc_mtx = get_rectangular_matrix_from_upper_triu(prcc_list[:self.upp_tri_size], self.n_ag)
        self.prcc_mtx = prcc_mtx
        p_icr = (1 - self.params['p']) * self.params['h'] * self.params['xi']
        self.p_icr = p_icr
        self.prcc_list = prcc_list
        return self.prcc_list

    def aggregate_lockdown_approaches(self, cm, agg_typ):
        agg_prcc = None
        if agg_typ == "simple":
            agg_prcc = np.sum(self.prcc_mtx, axis=1)
        elif agg_typ == 'relN':
            agg_prcc = np.sum(self.prcc_mtx * self.age_vector, axis=1) / np.sum(self.age_vector)
        elif agg_typ == 'relM':
            agg_prcc = (self.prcc_mtx @ self.age_vector) / np.sum(self.age_vector)
        elif agg_typ == 'cm':
            agg_prcc = np.sum(self.prcc_mtx * cm, axis=1)
        elif agg_typ == 'cmT':
            agg_prcc = np.sum(self.prcc_mtx * cm.T, axis=1)
        elif agg_typ == 'cmR':
            agg_prcc = np.sum(self.prcc_mtx * cm, axis=1) / np.sum(cm, axis=1)
        elif agg_typ == 'CMT':
            agg_prcc = np.sum((self.prcc_mtx * cm.T) / (np.sum(cm, axis=1)).T, axis=1)
        elif agg_typ == 'pval':
            self.calculate_p_values()
            agg_prcc = np.sum(self.p_value_mtx * self.prcc_mtx, axis=1)
        # save all the agg values from different approaches
        self.agg_prcc = agg_prcc
        return agg_prcc.flatten()

    def calculate_p_values(self):
        t = self.prcc_list * np.sqrt((self.number_of_samples - 2 - self.upp_tri_size) / (1 - self.prcc_list ** 2))
        # p-value for 2-sided test
        dof = self.number_of_samples - 2 - self.upp_tri_size
        p_value = 2 * (1 - ss.t.cdf(abs(t), dof))
        self.p_value = p_value

        prcc_p_value = get_rectangular_matrix_from_upper_triu(p_value[:self.upp_tri_size], self.n_ag)
        self.p_value_mtx = prcc_p_value
        return prcc_p_value

    def aggregate_p_values_approach(self, cm, agg_typ):
        # using the same approaches used to aggregate the prcc values, one best approach can be selected for both
        # most probably the same approach to aggregate both the prcc and p_values
        agg_pval = None
        if agg_typ == "simple":
            agg_pval = np.sum(self.p_value_mtx, axis=1)
        elif agg_typ == 'relN':
            agg_pval = np.sum(self.p_value_mtx * self.age_vector, axis=1) / np.sum(self.age_vector)
        elif agg_typ == 'relM':
            agg_pval = (self.p_value_mtx @ self.age_vector) / np.sum(self.age_vector)
        elif agg_typ == 'cm':
            agg_pval = np.sum(self.p_value_mtx * cm, axis=1)
        elif agg_typ == 'cmT':
            agg_pval = np.sum(self.p_value_mtx * cm.T, axis=1)
        elif agg_typ == 'cmR':
            agg_pval = np.sum(self.p_value_mtx * cm, axis=1) / np.sum(cm, axis=1)
        elif agg_typ == 'CMT':
            agg_pval = np.sum((self.p_value_mtx * cm.T) / (np.sum(cm, axis=1)).T, axis=1)
        elif agg_typ == 'pval':
            agg_pval = np.sum(self.p_value_mtx * self.prcc_mtx, axis=1)
        # save all the agg values from different approaches
        self.agg_pval = agg_pval
        return agg_pval.flatten()
