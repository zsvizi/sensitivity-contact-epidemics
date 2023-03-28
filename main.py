from src.dataloader import DataLoader
from src.simulation_npi import SimulationNPI


def main():
    data = DataLoader()
    simulation = SimulationNPI(data=data)
    simulation.generate_lhs()
    simulation.calculate_prcc_values()
    simulation.plot_prcc_values()


if __name__ == '__main__':
    main()
