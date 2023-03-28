import numpy as np
import scipy.stats as ss

from src.prcc import get_prcc_values, get_rectangular_matrix_from_upper_triu
from src.simulation_npi import SimulationNPI


class PRCCCalculator:
    def __init__(self, sim_state: dict, sim_obj: SimulationNPI,
                 number_of_samples: int):

        self.n_ag = sim_obj.n_ag
        self.age_vector = sim_obj.age_vector
        self.params = sim_obj.params
        self.number_of_samples = number_of_samples

        self.upp_tri_size = int((self.n_ag + 1) * self.n_ag / 2)
        self.sim_state = sim_state
        self.base_r0 = sim_state["base_r0"]

        self.prcc_mtx = np.array([])
        self.p_icr = []

        self.prcc_matrix_school = []  # 16 * 16
        self.prcc_matrix_work = []
        self.prcc_matrix_other = []
        self.prcc_list = None

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

    def aggregate_approach(self):
        agg_prcc_school, agg_prcc_work, agg_prcc_other = [
            (np.sum(self.prcc_matrix_school * self.age_vector, axis=1) /
             np.sum(self.age_vector)).flatten(),
            (np.sum(self.prcc_matrix_work * self.age_vector, axis=1) /
             np.sum(self.age_vector)).flatten(),
            (np.sum(self.prcc_matrix_other * self.age_vector, axis=1) /
             np.sum(self.age_vector)).flatten()]

        # Save aggregated prcc values
        np.savetxt("sens_data/school_prcc_values.csv", agg_prcc_school, delimiter=",")
        np.savetxt("sens_data/work_prcc_values.csv", agg_prcc_work, delimiter=",")
        np.savetxt("sens_data/other_prcc_values.csv", agg_prcc_other, delimiter=",")

    def calculate_p_values(self,  mtx_typ):
        t = self.prcc_list * np.sqrt((self.number_of_samples - 2 - self.upp_tri_size) / (1 - self.prcc_list ** 2))
        # p-value for 2-sided test
        dof = self.number_of_samples - 2 - self.upp_tri_size
        p_value = 2 * (1 - ss.t.cdf(abs(t), dof))
        p_value_school = p_value[:self.upp_tri_size]
        p_value_work = p_value[self.upp_tri_size: 2 * self.upp_tri_size]
        p_value_other = p_value[2 * self.upp_tri_size:]
        print("s", p_value_school)
        print("w", p_value_work)

        # aggregate using p-values. First get the p-values in matrix form: 16 * 16
        if mtx_typ == "lockdown_3":
            school = get_rectangular_matrix_from_upper_triu(p_value[:self.upp_tri_size], self.n_ag)
            work = get_rectangular_matrix_from_upper_triu(p_value[self.upp_tri_size:2 * self.upp_tri_size], self.n_ag)
            other = get_rectangular_matrix_from_upper_triu(p_value[2 * self.upp_tri_size:], self.n_ag)

            p_agg_school = np.sum(school * self.prcc_matrix_school, axis=1)
            p_agg_work = np.sum(work * self.prcc_matrix_work, axis=1)
            p_agg_other = np.sum(other * self.prcc_matrix_other, axis=1)
        elif mtx_typ == "lockdown":
            prcc_p_value = get_rectangular_matrix_from_upper_triu(p_value[:self.upp_tri_size], self.n_ag)
