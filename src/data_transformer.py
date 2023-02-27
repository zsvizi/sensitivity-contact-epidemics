import numpy as np
from src.prcc_calculation import PRCCalculator
from src.dataloader import DataLoader
from src.model import RostModelHungary
from src.r0generator import R0Generator


class Transformer:
    def __init__(self):
        self.data = DataLoader()

        self.sim_state = dict()
        self.sim_obj = dict()

        self.n_ag = self.data.contact_data["home"].shape[0]
        self.model = RostModelHungary(model_data=self.data)
        self.population = self.model.population
        self.age_vector = self.population.reshape((-1, 1))
        self.susceptibles = self.model.get_initial_values()[self.model.c_idx["s"] *
                                                            self.n_ag:(self.model.c_idx["s"] + 1) * self.n_ag]
        self.upper_tri_indexes = np.triu_indices(self.n_ag)
        # 0. Get base parameter dictionary
        self.params = self.data.model_parameters_data
        self.contact_matrix = self.data.contact_data["home"] + self.data.contact_data["work"] + \
            self.data.contact_data["school"] + self.data.contact_data["other"]
        self.contact_home = self.data.contact_data["home"]

        self.prcc_values = PRCCalculator(age_vector=self.age_vector, params=self.params, n_ag=self.n_ag,
                                         model=self.model)

        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.2, 1.8, 2.5]
        self.mtx_types = ["lockdown", "lockdown3", "mitigation"]

        self.get_data_sensitivity()

    def get_data_sensitivity(self):
        self.sim_obj.update({
            "no_age": self.n_ag,
            "model": self.model,
            "population": self.population,
            "age_vector": self.age_vector,
            "contact_matrix": self.contact_matrix,
            "contact_home": self.contact_home,
            "susceptibles": self.susceptibles,
            "upper_tri_indexes": self.upper_tri_indexes,
            "params": self.params
         })

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
                    self.sim_state.update({"base_r0": base_r0,
                                           "beta": beta,
                                           "type": mtx_type,
                                           "susc": susc,
                                           "r0generator": r0generator})
