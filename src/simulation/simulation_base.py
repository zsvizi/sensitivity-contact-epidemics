from abc import ABC, abstractmethod
import numpy as np
from src.dataloader import DataLoader
from src.model.model_base import EpidemicModelBase


class SimulationBase(ABC):
    """
    Abstract base class for epidemiological simulation frameworks.

    This class provides the core setup for epidemic simulations, including:
      - Loading model and contact data
      - Managing country-specific contact matrices
      - Initializing population and parameter structures
    """

    def __init__(self, data: DataLoader, country: str):
        """
        Initializes the simulation environment and load all required model data.

        :param DataLoader data: DataLoader object providing epidemiological parameters,
                                contact matrices, and population data.
        :param str country: Country identifier used to select the correct contact matrix set.
        """
        self.data = data
        self.sim_state = dict()
        self.model = EpidemicModelBase(model_data=data)

        # Attributes set during initialization
        self.contact_matrix: np.ndarray
        self.contact_home: np.ndarray
        self.n_ag: int

        # Load appropriate contact matrices for the selected country
        self.__set_contact_data(country=country)

        # Population and age-related data
        self.population = self.model.population
        self.age_vector = self.population.reshape((-1, 1))

        # Indexing and sizing attributes for later usage
        self.upper_tri_indexes = np.triu_indices(self.n_ag)
        self.params = self.data.model_parameters_data
        self.upper_tri_size = int((self.n_ag + 1) * self.n_ag / 2)

    def __set_contact_data(self, country: str) -> None:
        """
        Configures contact matrices and related demographic data based on the selected country.

        :param str country: The name of the country whose contact data should be loaded
        """
        if country in ["None", "Hungary_maszk", "united_states"]:
            self.contact_matrix = self.data.contact_data["All"]
            self.contact_home = self.data.contact_data["Home"]
            self.n_ag = self.data.contact_data["Home"].shape[0]
        elif country == "UK":
            self.contact_matrix = self.data.contact_data["All"]
            self.contact_home = self.data.contact_data["Physical"]
            self.n_ag = self.data.contact_data["All"].shape[0]
        else:
            self.contact_matrix = (
                    self.data.contact_data["Home"] +
                    self.data.contact_data["School"] +
                    self.data.contact_data["Work"] +
                    self.data.contact_data["Other"]
            )
            self.contact_home = self.data.contact_data["Home"]
            self.n_ag = self.data.contact_data["Home"].shape[0]

    @abstractmethod
    def _choose_model(self, epi_model: str):
        """
        Abstract method to be implemented by subclass (SimulationNPI), defining the epidemiological model to use.
        This method should initialize and assign a model object.

        :param str epi_model: The name of the epidemiological model to use.
        """
        pass
