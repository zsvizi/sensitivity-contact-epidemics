import numpy as np
from src.analysis_npi import AnalysisNPI
from src.dataloader import DataLoader
from src.r0generator import R0Generator
from src.sampler_npi import SamplerNPI
from src.prcc_calculation import PRCCalculator


class SimulationNPI:
    def __init__(self, sim_state, sim_obj):
        # Load data
        self.data = DataLoader()
        self.sim_state = sim_state
        self.sim_obj = sim_obj

        # User-defined parameters
        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.2, 1.8, 2.5]

        self.mtx_types = ["lockdown", "mitigation", "lockdown_3"]

    def generate_lhs(self):
        # 1. Update params by susceptibility vector
        susceptibility = np.ones(self.sim_obj.n_ag)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            self.sim_obj.params.update({"susc": susceptibility})
            # 2. Update params by calculated BASELINE beta
            for base_r0 in self.r0_choices:
                r0generator = R0Generator(param=self.sim_obj.params)
                beta = base_r0 / r0generator.get_eig_val(contact_mtx=self.sim_obj.contact_matrix,
                                                         susceptibles=self.sim_obj.susceptibles.reshape(1, -1),
                                                         population=self.sim_obj.population)[0]
                self.sim_obj.params.update({"beta": beta})
                # 3. Choose matrix type
                for mtx_type in self.mtx_types:
                    sim_state = {"base_r0": base_r0, "beta": beta, "type": mtx_type, "susc": susc,
                                 "r0generator": r0generator}
                    self.sim_state = sim_state

                    cm_generator = SamplerNPI(sim_state=sim_state, sim_obj=self,
                                              get_output=self.sim_obj.contact_matrix,
                                              get_sim_output=self.sim_obj.contact_matrix)
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
        susceptibility = np.ones(self.sim_obj.n_ag)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            for base_r0 in self.r0_choices:
                print(base_r0)
                PRCCalculator.calculate_prcc_values(self.sim_obj.contact_matrix)

    def _get_upper_bound_factor_unit(self):
        cm_diff = (self.sim_obj.contact_matrix - self.sim_obj.contact_home) * self.sim_obj.age_vector
        min_diff = np.min(cm_diff) / 2
        return min_diff
