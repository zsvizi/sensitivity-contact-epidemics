import numpy as np

from src.dataloader import DataLoader
from src.model import RostModelHungary


class SimulationBase:
    def __init__(self):
        self.data = DataLoader()

        self.base_r0 = 2.2
        self.beta = float()
        self.sim_state = dict()

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

        self.kappas = [0.789078907890789, 0.43344334433443343, 0.22882288228822883,
                       0.7817781778177818, 0.41324132413241327, 0.20032003200320034,
                       0.789078907890789, 0.43344334433443343, 0.22882288228822883,
                       0.7817781778177818, 0.41324132413241327, 0.20032003200320034,
                       0.789078907890789, 0.43344334433443343, 0.22882288228822883,
                       0.7817781778177818
                       ]
