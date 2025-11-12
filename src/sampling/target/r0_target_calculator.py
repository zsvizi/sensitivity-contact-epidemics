import numpy as np

import src
from src.examples.rost.r0 import R0Generator
from src.sampling.target.target_calculator import TargetCalculator
from src.examples.seir.r0 import R0SeirSVModel
from src.examples.chikina.r0 import R0SirModel
from src.examples.moghadas.r0 import R0SeyedModel
from src.examples.rost.r0 import R0Generator
from src.examples.validation.r0 import R0ValidationModel
from src.sampling.target.target_calculator import TargetCalculator


class R0TargetCalculator(TargetCalculator):
    """
    Target calculator for computing the basic reproduction number (R_0)
    under different contact matrix configurations.

    This class selects the appropriate R_0 generator depending on the
    simulation's target country and model setup. It then computes the
    effective R_0 by combining the base transmission rate (beta) with the
    dominant eigenvalue of the next-generation matrix.
    """

    def __init__(self, sim_obj: src.SimulationNPI):
        """
        Initializes the R_§ target calculator.

        :param src.SimulationNPI sim_obj: Simulation object containing
                                          model state, country, and parameters.
        """
        self.country = sim_obj.country
        super().__init__(sim_obj=sim_obj)
        self.base_r0 = sim_obj.sim_state["base_r0"]
        self.beta = sim_obj.sim_state["beta"]

    def get_output(self, cm: np.ndarray) -> np.ndarray:
        """
        Compute the effective reproduction number (R_0) for a given contact matrix.

        Depending on the simulation country, a specific epidemiological model
        is used to compute the eigenvalue of the next-generation matrix (NGM).
        The effective R_0 is then calculated as:
            (R_0)_{effective} = beta * lambda_max(NGM)

        :param np.ndarray cm: Contact matrix representing interactions.
        :return np.ndarray: array containing the computed effective R_0 value.
        """
        # Select appropriate R_0 generator based on country (model type)
        if self.country in ["Hungary_maszk", "Hungary_prem"]:
            r0generator = R0Generator(param=self.sim_obj.params,
                                      n_age=self.sim_obj.n_ag)
        elif self.country == "UK":
            r0generator = R0SeirSVModel(param=self.sim_obj.params)
        elif self.country == "usa":
            r0generator = R0SirModel(param=self.sim_obj.params)
        elif self.country == "united_states":
            r0generator = R0SeyedModel(param=self.sim_obj.params)
        elif self.country == "None":
            r0generator = R0ValidationModel(param=self.sim_obj.params)
        else:
            raise Exception("Invalid country! Cannot select R₀ generator.")

        # Compute dominant eigenvalue of the next-generation matrix
        r0_lhs = self.beta * r0generator.get_eig_val(
            contact_mtx=cm,
            susceptibles=self.sim_obj.susceptibles.reshape(1, -1),
            population=self.sim_obj.population
        )[0]

        # Return the R_0 value as a NumPy array for downstream processing
        output = np.array([r0_lhs])
        return output
