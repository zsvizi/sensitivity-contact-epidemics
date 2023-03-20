import numpy as np
from src.simulation_base import SimulationBase
from src.r0_generator import R0Generator
from src.sampler_npi import SamplerNPI
from src.prcc_calculation import PRCCCalculator


class SimulationNPI(SimulationBase):
    def __init__(self) -> None:
        super().__init__()

        # User-defined parameters
        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.2, 1.8, 2.5]

        self.mtx_types = ["lockdown", "lockdown_3"]

    def generate_lhs(self):
        # 1. Update params by susceptibility vector
        susceptibility = np.ones(16)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            self.params.update({"susc": self.susceptibles})
            self.sim_state.update({"susc": susceptibility})
            # 2. Update params by calculated BASELINE beta
            for base_r0 in self.r0_choices:
                r0generator = R0Generator(param=self.params)
                beta = base_r0 / r0generator.get_eig_val(contact_mtx=self.contact_matrix,
                                                         susceptibles=self.susceptibles.reshape(1, -1),
                                                         population=self.population)[0]
                self.params.update({"beta": beta})
                # 3. Choose matrix type
                for mtx_type in self.mtx_types:
                    self.sim_state.update(
                        {"base_r0": base_r0,
                         "beta": beta,
                         "type": mtx_type,
                         "susc": susc,
                         "r0generator": r0generator})
                    cm_generator = SamplerNPI(sim_state=self.sim_state, data_tr=self, mtx_type=mtx_type)
                    lhs_table, sim_output = cm_generator.run()
                    prcc_calculator = PRCCCalculator(age_vector=self.age_vector,
                                                     params=self.params, n_ag=self.n_ag, data_tr=self,
                                                     sim_state=self.sim_state)
                    prcc_calculator.calculate_prcc_values(mtx_typ=mtx_type, lhs_table=lhs_table, sim_output=sim_output)
                    prcc_calculator.aggregate_approach()
                    prcc_calculator.calculate_p_values(mtx_typ=mtx_type, lhs_table=lhs_table, sim_output=sim_output)

    def _get_upper_bound_factor_unit(self):
        cm_diff = (self.contact_matrix - self.contact_home) * self.age_vector
        min_diff = np.min(cm_diff) / 2
        return min_diff
