import numpy as np

from analysis import Analysis
from dataloader import DataLoader
from model import RostModelHungary
from plotter import generate_prcc_plots, generate_stacked_plots
from r0 import R0Generator
from sampler import ContactMatrixSampler


class Simulation:
    def __init__(self):
        # Load data
        self.data = DataLoader()

        # User-defined parameters
        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.35, 1.1, 1.6, 2.5]

        # Define initial configs
        self._get_initial_config()

        # For contact matrix sampling: ["unit", "ratio"]
        self.mtx_types = ["lockdown", "ratio", "mitigation"]

    def run(self):
        is_lhs_generated = True
        is_prcc_plots_generated = False

        # 1. Update params by susceptibility vector
        susceptibility = np.ones(self.no_ag)
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
                    sim_state = {"base_r0": base_r0, "beta": beta, "mtx_type": mtx_type, "susc": susc,
                                 "r0generator": r0generator}
                    if is_lhs_generated:
                        cm_generator = ContactMatrixSampler(sim_state=sim_state, sim_obj=self)
                        cm_generator.run()

                    else:
                        pass
                        # if susc in [1.0, 0.5] and base_r0 in [1.35] and mtx_type in ["lockdown"]:
                        #     analysis = Analysis(sim=self, susc=susc, base_r0=base_r0, mtx_type=mtx_type)
                        #     analysis.run()
        if is_prcc_plots_generated:
            generate_prcc_plots(sim_obj=self)

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


def main():
    simulation = Simulation()
    simulation.run()


if __name__ == '__main__':
    # generate_stacked_plots()
    main()

