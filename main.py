from src.simulation_npi import SimulationNPI
from src.plotter import Plotter
from src.dataloader import DataLoader
from src.prcc_calculation import PRCCCalculator


def main():
    data = DataLoader()

    simulation = SimulationNPI()
    prcc = PRCCCalculator(age_vector=simulation.age_vector, params=simulation.params, n_ag=simulation.n_ag)
    simulation.generate_lhs()
    simulation.generate_prcc_values()
    Plotter.plot_2d_contact_matrices(self=data)


if __name__ == '__main__':
    main()
