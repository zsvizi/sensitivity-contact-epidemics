import numpy as np

from analysis_npi import AnalysisNPI
from dataloader import DataLoader
from model import RostModelHungary
from r0 import R0Generator
from sampler_npi import NPISampler
from PRCC_calculation import PRCCalculator


class SimulationNPI:
    def __init__(self, sim_state):
        # Load data
        self.data = DataLoader()
        self.sim_state = sim_state

        # User-defined parameters
        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.2, 1.8, 2.5]

        # Define initial configs
        self._get_initial_config()

        # For contact matrix sampling: ["lockdown", "mitigation", "lockdown_3"]
        # self.mtx_types = ["lockdown", "mitigation", "lockdown_3"]
        self.mtx_types = ["lockdown"]

    def generate_lhs(self):
        # 1. Update params by susceptibility vector
        susceptibility = np.ones(self.no_ag)
        i = 0
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            self.params.update({"susc": susceptibility})
            # 2. Update params by calculated BASELINE beta
            for base_r0 in self.r0_choices:
                r0generator = R0Generator(param=self.params)
                beta = base_r0 / r0generator.get_eig_val(contact_mtx=self.contact_matrix,
                                                         susceptibles=self.susceptibles.reshape(1, -1),
                                                         population=self.population)[0]
                self.params.update({"beta": beta})
                # 3. Choose matrix type
                for mtx_type in self.mtx_types:
                    sim_state = {"base_r0": base_r0, "beta": beta, "type": mtx_type, "susc": susc,
                                 "r0generator": r0generator}
                    self.sim_state = sim_state

                    cm_generator = NPISampler(sim_state=sim_state, sim_obj=self, cm_entries=self.contact_matrix,
                                              entries_lockdown=self.contact_matrix,
                                              entries_lockdown3=self.contact_matrix,
                                              get_output=self.contact_matrix)
                    cm_generator.run()

    def get_analysis_results(self):
        i = 0
        for susc in self.susc_choices:
            for base_r0 in self.r0_choices:
                kappas = [0.789078907890789, 0.43344334433443343, 0.22882288228822883,
                          0.7817781778177818, 0.41324132413241327, 0.20032003200320034]
                print(susc, base_r0)
                analysis = AnalysisNPI(sim=self, susc=susc, base_r0=base_r0, kappa=kappas[i])
                analysis.run()
                i += 1

    def prcc_plots_generation(self):
        susceptibility = np.ones(self.no_ag)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            for base_r0 in self.r0_choices:
                print(base_r0)
                PRCCalculator.calculate_prcc_values(self.contact_matrix)

    def _get_initial_config(self):
        self.no_ag = self.data.contact_data["home"].shape[0]
        self.model = RostModelHungary(model_data=self.data)
        self.population = self.model.population
        self.age_vector = self.population.reshape((-1, 1))
        self.susceptibles = self.model.get_initial_values()[self.model.c_idx["s"] *
                                                            self.no_ag:(self.model.c_idx["s"] + 1) * self.no_ag]
        self.contact_matrix = self.data.contact_data["home"] + self.data.contact_data["work"] + \
            self.data.contact_data["school"] + self.data.contact_data["other"]
        self.contact_home = self.data.contact_data["home"]
        self.upper_tri_indexes = np.triu_indices(self.no_ag)
        # 0. Get base parameter dictionary
        self.params = self.data.model_parameters_data

    def _get_upper_bound_factor_unit(self):
        cm_diff = (self.contact_matrix - self.contact_home) * self.age_vector
        min_diff = np.min(cm_diff) / 2
        return min_diff
