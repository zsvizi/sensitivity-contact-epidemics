import numpy as np

from src.seirsv.dataloader import seirSVDataLoader
from src.seirsv.model_seirsv import SeirSVModel


class SimulationBase:
    def __init__(self, data: seirSVDataLoader):
        self.data = data
        self.sim_state = dict()
        self.n_ag = self.data.contact_data["all_contact"].shape[0]
        self.physical_contact = self.data.contact_data["physical_contact"]
        self.all_contact = self.data.contact_data["all_contact"]

        self.model = SeirSVModel(model_data=self.data)

        self.population = self.model.population
        self.age_vector = self.population.reshape((-1, 1))
        self.susceptibles = self.model.get_initial_values()[self.model.c_idx["s"] *
                                                            self.n_ag:(self.model.c_idx["s"] + 1) *
                                                                      self.n_ag]

        self.upper_tri_indexes = np.triu_indices(self.n_ag)
        # 0. Get base parameter dictionary
        self.params = self.data.model_parameters_data

        self.upper_tri_size = int((self.n_ag + 1) * self.n_ag / 2)
