from abc import ABC, abstractmethod

import numpy as np

from src.dataloader import DataLoader
from src.model.model_base import EpidemicModelBase


class SimulationBase(ABC):
    def __init__(self, data: DataLoader, country: str):
        self.data = data

        self.sim_state = dict()
        self.model = EpidemicModelBase(model_data=data)

        self.contact_matrix: np.array
        self.contact_home: np.array
        self.n_ag: int

        self.__set_contact_data(country=country)

        self.population = self.model.population
        self.age_vector = self.population.reshape((-1, 1))

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

    @abstractmethod
    def _choose_model(self, epi_model):
        return
