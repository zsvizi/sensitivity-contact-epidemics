import numpy as np

from src.dataloader import DataLoader
from src.model.model import RostModelHungary
from src.model.model_sir import SirModel


class SimulationBase:
    def __init__(self, data: DataLoader, epi_model):
        self.data = data
        self.sim_state = dict()

        self.n_ag = self.data.contact_data["Home"].shape[0]
        self.contact_matrix = self.data.contact_data["Home"] + self.data.contact_data["Work"] + \
            self.data.contact_data["School"] + self.data.contact_data["Other"]
        self.contact_home = self.data.contact_data["Home"]
        if epi_model == "rost_model":
            self.model = RostModelHungary(model_data=self.data)
        elif epi_model == "sir_model":
            self.model = SirModel(model_data=self.data)
        else:
            raise Exception("No model was given!")

        self.population = self.model.population
        self.age_vector = self.population.reshape((-1, 1))
        self.susceptibles = self.model.get_initial_values()[self.model.c_idx["s"] *
                                                            self.n_ag:(self.model.c_idx["s"] + 1) * self.n_ag]

        self.upper_tri_indexes = np.triu_indices(self.n_ag)
        # 0. Get base parameter dictionary
        self.params = self.data.model_parameters_data
        self.upper_tri_size = int((self.n_ag + 1) * self.n_ag / 2)
