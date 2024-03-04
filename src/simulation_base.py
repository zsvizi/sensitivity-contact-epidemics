import numpy as np

from src.dataloader import DataLoader
from src.model.model_base import EpidemicModelBase
from src.model.model import RostModelHungary
from src.chikina.model import SirModel
from src.moghadas.model import MoghadasModelUsa
from src.seir.model import SEIR_UK


class SimulationBase:
    def __init__(self, data: DataLoader, epi_model: str, country: str):
        self.data = data
        self.sim_state = dict()
        self.model: EpidemicModelBase
        self.contact_matrix: np.array
        self.contact_home: np.array
        self.n_ag: int

        self.__set_contact_data(country=country)
        self.__choose_model(epi_model=epi_model)

        self.population = self.model.population
        self.age_vector = self.population.reshape((-1, 1))
        self.susceptibles = self.model.get_initial_values()[self.model.c_idx["s"] *
                                                            self.n_ag:(self.model.c_idx["s"] + 1) * self.n_ag]

        self.upper_tri_indexes = np.triu_indices(self.n_ag)
        # 0. Get base parameter dictionary
        self.params = self.data.model_parameters_data
        self.upper_tri_size = int((self.n_ag + 1) * self.n_ag / 2)

    def __set_contact_data(self, country):
        if country == "united_states":
            self.contact_matrix = self.data.contact_data["All"]
            self.contact_home = self.data.contact_data["Home"]
            self.n_ag = self.data.contact_data["Home"].shape[0]
        elif country == "UK":
            self.contact_matrix = self.data.contact_data["All"]
            self.contact_home = self.data.contact_data["Physical"]
            self.n_ag = self.data.contact_data["All"].shape[0]
        else:
            self.contact_matrix = self.data.contact_data["Home"] + \
                                  self.data.contact_data["School"] + \
                                  self.data.contact_data["Work"] + \
                                  self.data.contact_data["Other"]
            self.contact_home = self.data.contact_data["Home"]
            self.n_ag = self.data.contact_data["Home"].shape[0]

    def __choose_model(self, epi_model):
        if epi_model == "rost":
            self.model = RostModelHungary(model_data=self.data)
        elif epi_model == "chikina":
            self.model = SirModel(model_data=self.data)
        elif epi_model == "seir":
            self.model = SEIR_UK(model_data=self.data)
        elif epi_model == "moghadas":
            self.model = MoghadasModelUsa(model_data=self.data)
        else:
            raise Exception("No model was given!")
