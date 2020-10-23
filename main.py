import os

import matplotlib.pyplot as plt
import numpy as np

from dataloader import DataLoader
from model import RostModelHungary
from r0 import R0Generator


def main():
    data = DataLoader()
    # Define initial configs
    susc_choices = [0.1, 0.5, 1.0]
    r0_choices = [0.9, 1.65, 2.2, 3]
    lower_matrix_types = ["home", "normed"]

    # Get population age vector
    age_vector = data.age_data.reshape((-1, 1))
    # Get contact matrices
    contact_matrix = \
        data.contact_data["home"] + data.contact_data["work"] + \
        data.contact_data["school"] + data.contact_data["other"]
    contact_home = data.contact_data["home"]
    upper_limit_matrix = contact_matrix * age_vector
    lower_limit_matrices = {"normed": upper_limit_matrix - np.min(upper_limit_matrix - contact_home * age_vector),
                            "home": contact_home * age_vector}
    upper_tri_indexes = np.triu_indices(data.contact_data["home"].shape[0])

    # Get model and initial values
    model = RostModelHungary(model_data=data)
    s_idx = model.c_idx["s"]
    susceptibles = model.get_initial_values()[s_idx * 16:(s_idx + 1) * 16]
    population = model.population

    # Get base parameter dictionary
    params = data.model_parameters_data

    # Update params by susceptibility vector
    susceptibility = np.ones(16)
    susceptibility[:4] = susc_choices[-1]
    params.update({"susc": susceptibility})

    # Update params by calculated beta
    base_r0 = r0_choices[0]
    r0generator = R0Generator(param=params)
    beta = base_r0 / r0generator.get_eig_val(contact_mtx=contact_matrix,
                                             susceptibles=susceptibles.reshape(1, -1),
                                             population=population)[0]
    params.update({"beta": beta})

    # Get solution
    t = np.linspace(0, 500, 1000)
    solution = model.get_solution(t=t, parameters=params, cm=contact_matrix)

    # Plot solution
    plt.plot(t, solution[:, s_idx * 16:(s_idx + 1) * 16])
    plt.show()


if __name__ == "__main__":
    main()
