import numpy as np

from src.dataloader import DataLoader
from src.model import RostModelHungary
from src.r0_generator import R0Generator


class DataTransformer:
    def __init__(self):
        self.data = DataLoader()

        self.base_r0 = float()
        self.beta = float()

        self.sim_state_data = dict()

        self.n_ag = self.data.contact_data["home"].shape[0]
        self.model = RostModelHungary(model_data=self.data)
        self.population = self.model.population
        self.age_vector = self.population.reshape((-1, 1))
        self.susceptibles = self.model.get_initial_values()[self.model.c_idx["s"] *
                                                            self.n_ag:(self.model.c_idx["s"] + 1) * self.n_ag]
        self.upper_tri_indexes = np.triu_indices(self.n_ag)
        # 0. Get base parameter dictionary
        self.params = self.data.model_parameters_data
        self.upper_tri_size = int((self.n_ag + 1) * self.n_ag / 2)

        self.contact_matrix = self.data.contact_data["home"] + self.data.contact_data["work"] + \
            self.data.contact_data["school"] + self.data.contact_data["other"]
        self.contact_home = self.data.contact_data["home"]

        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.2, 1.8, 2.5]
        self.mtx_types = ["lockdown", "lockdown3", "mitigation"]
        self.kappas = [0.789078907890789, 0.43344334433443343, 0.22882288228822883,
                       0.7817781778177818, 0.41324132413241327, 0.20032003200320034]

        self.get_data_sensitivity()

    def get_data_sensitivity(self):

        susceptibility = np.ones(self.n_ag)
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
                    self.sim_state_data.update({"base_r0": base_r0, "beta": beta, "type": mtx_type,
                                                "susc": susc, "r0generator": r0generator})
                    self.beta = beta
                    self.base_r0 = base_r0
