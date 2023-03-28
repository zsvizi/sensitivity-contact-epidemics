
from src.simulation_npi import SimulationNPI
from src.dataloader import DataLoader


def main():

    simulation = SimulationNPI()
    simulation.generate_lhs()


if __name__ == '__main__':
    main()
