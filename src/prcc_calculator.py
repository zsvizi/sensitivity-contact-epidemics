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

    def calculate_prcc_values(self, mtx_typ, lhs_table: np.ndarray, sim_output: np.ndarray):
        if "lockdown_3" == mtx_typ:
            sim_data = lhs_table[:, :3 * self.upp_tri_size]
            sim_data = 1 - sim_data
        elif "lockdown" == mtx_typ:
            sim_data = lhs_table[:, :(self.n_ag * (self.n_ag + 1)) // 2]
            sim_data = 1 - sim_data
        else:
            raise Exception('Matrix type is unknown!')
        simulation = np.append(sim_data, sim_output[:, -self.n_ag - 1].reshape((-1, 1)), axis=1)
        prcc_list = get_prcc_values(simulation, number_of_samples=self.number_of_samples)
        if "lockdown_3" == mtx_typ:
            prcc_matrix_school = get_rectangular_matrix_from_upper_triu(
                prcc_list[:self.upp_tri_size], self.n_ag)
            prcc_matrix_work = get_rectangular_matrix_from_upper_triu(
                prcc_list[self.upp_tri_size:2 * self.upp_tri_size], self.n_ag)
            prcc_matrix_other = get_rectangular_matrix_from_upper_triu(
                prcc_list[2 * self.upp_tri_size:], self.n_ag)
            self.prcc_matrix_school = prcc_matrix_school
            self.prcc_matrix_work = prcc_matrix_work
            self.prcc_matrix_other = prcc_matrix_other
        elif "lockdown" == mtx_typ:
            prcc_mtx = get_rectangular_matrix_from_upper_triu(prcc_list[:self.upp_tri_size], self.n_ag)
            self.prcc_mtx = prcc_mtx
            p_icr = (1 - self.params['p']) * self.params['h'] * self.params['xi']
            self.p_icr = p_icr
        self.prcc_list = prcc_list
        return self.prcc_list

    def aggregate_lockdown_approaches(self, cm, mtx_typ, agg_typ):
        if "lockdown" == mtx_typ:

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
                self.calculate_p_values(mtx_typ=mtx_typ)
                agg_prcc = np.sum(self.p_value_mtx * self.prcc_mtx, axis=1)
            # save all the agg values from different approaches
            self.agg_prcc = agg_prcc
            return agg_prcc.flatten()

    def aggregate_lockdown_3_approach(self, cm, mtx_typ, agg_typ):
        if agg_typ == "simple":
            agg_prcc_school, agg_prcc_work, agg_prcc_other = [
                (np.sum(self.prcc_matrix_school * self.age_vector, axis=1) /
                 np.sum(self.age_vector)).flatten(),
                (np.sum(self.prcc_matrix_work * self.age_vector, axis=1) /
                 np.sum(self.age_vector)).flatten(),
                (np.sum(self.prcc_matrix_other * self.age_vector, axis=1) /
                 np.sum(self.age_vector)).flatten()]
            prcc_list = np.array([agg_prcc_school, agg_prcc_work, agg_prcc_other]).flatten()   # size 48
        elif agg_typ == 'cm':
            cm_school, cm_work, cm_other = [
                (np.sum(self.prcc_matrix_school * cm, axis=1) /
                 np.sum(self.age_vector)).flatten(),
                (np.sum(self.prcc_matrix_work * cm, axis=1) /
                 np.sum(self.age_vector)).flatten(),
                (np.sum(self.prcc_matrix_other * cm, axis=1) /
                 np.sum(self.age_vector)).flatten()]
            prcc_list = np.array([cm_school, cm_work, cm_other]).flatten()  # size 48
        elif agg_typ == 'simcm':
            simcm_school, simcm_work, simcm_other = [
                np.sum(self.prcc_matrix_school * cm, axis=1),
                np.sum(self.prcc_matrix_work * cm, axis=1),
                np.sum(self.prcc_matrix_other * cm, axis=1)]
            prcc_list = np.array([simcm_school, simcm_work, simcm_other]).flatten()  # size 48
        elif agg_typ == 'pval':
            if mtx_typ == "lockdown_3":
                school, work, other = self.calculate_p_values(mtx_typ=mtx_typ)  # get calculated lockdown_3 p values
                p_agg_school = np.sum(school * self.prcc_matrix_school, axis=1)
                p_agg_work = np.sum(work * self.prcc_matrix_work, axis=1)
                p_agg_other = np.sum(other * self.prcc_matrix_other, axis=1)
                prcc_list = np.array([p_agg_school, p_agg_work, p_agg_other]).flatten()
        # save all the aggregated values from the different approaches
        self.agg_lock3 = prcc_list

    def calculate_p_values(self, mtx_typ):
        t = self.prcc_list * np.sqrt((self.number_of_samples - 2 - self.upp_tri_size) / (1 - self.prcc_list ** 2))
        # p-value for 2-sided test
        dof = self.number_of_samples - 2 - self.upp_tri_size
        p_value = 2 * (1 - ss.t.cdf(abs(t), dof))
        self.p_value = p_value
        p_value_school = p_value[:self.upp_tri_size]
        p_value_work = p_value[self.upp_tri_size: 2 * self.upp_tri_size]
        p_value_other = p_value[2 * self.upp_tri_size:]

        # get the p-values in matrix form: 16 * 16 to be used in aggregation
        if mtx_typ == "lockdown_3":
            school = get_rectangular_matrix_from_upper_triu(p_value[:self.upp_tri_size], self.n_ag)
            work = get_rectangular_matrix_from_upper_triu(p_value[self.upp_tri_size:2 * self.upp_tri_size],
                                                          self.n_ag)
            other = get_rectangular_matrix_from_upper_triu(p_value[2 * self.upp_tri_size:], self.n_ag)
            return school, work, other
        elif mtx_typ == "lockdown":
            prcc_p_value = get_rectangular_matrix_from_upper_triu(p_value[:self.upp_tri_size], self.n_ag)
            self.p_value_mtx = prcc_p_value
            return prcc_p_value

    def aggregate_p_values_approach(self, cm, mtx_typ, agg_typ):
        # using the same approaches used to aggregate the prcc values, one best approach can be selected for both
        # most probably the same approach to aggregate both the prcc and p_values
        if mtx_typ == "lockdown":
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

