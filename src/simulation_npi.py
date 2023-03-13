import numpy as np

from src.data_transformer import DataTransformer
from src.r0_generator import R0Generator
from src.sampler_npi import SamplerNPI
from src.prcc_calculation import PRCCCalculator


class SimulationNPI:
    def __init__(self, sim_state: SamplerNPI, data_tr: DataTransformer) -> None:
        # Load data
        self.sim_state = sim_state
        self.data_tr = data_tr

        # User-defined parameters
        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.2, 1.8, 2.5]

        self.mtx_types = ["lockdown", "mitigation", "lockdown_3"]

    def generate_lhs(self):
        # 1. Update params by susceptibility vector
        susceptibility = np.ones(16)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            self.data_tr.params.update({"susc": self.data_tr.susceptibles})
            self.sim_state.update({"susc": susceptibility})
            # 2. Update params by calculated BASELINE beta
            for base_r0 in self.r0_choices:
                r0generator = R0Generator(param=self.data_tr.params)
                beta = base_r0 / r0generator.get_eig_val(contact_mtx=self.data_tr.contact_matrix,
                                                         susceptibles=self.data_tr.susceptibles.reshape(1, -1),
                                                         population=self.data_tr.population)[0]
                self.data_tr.params.update({"beta": beta})
                # 3. Choose matrix type
                for mtx_type in self.mtx_types:
                    sim_state = {"base_r0": base_r0, "beta": beta, "type": mtx_type, "susc": susc,
                                 "r0generator": r0generator}
                    self.sim_state = sim_state
                    cm_generator = SamplerNPI(sim_state=sim_state, data_tr=self.data_tr)
                    cm_generator.run()

    def generate_prcc_values(self):
        susceptibility = np.ones(self.data_tr.n_ag)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            for base_r0 in self.r0_choices:
                print(base_r0)
                PRCCCalculator.calculate_prcc_values(self=self.data_tr.contact_matrix)

    def _get_upper_bound_factor_unit(self):
        cm_diff = (self.data_tr.contact_matrix - self.data_tr.contact_home) * self.data_tr.age_vector
        min_diff = np.min(cm_diff) / 2
        return min_diff
