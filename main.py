import src
from src.dataloader import DataLoader


def main():
    data = DataLoader()
    simulation = src.SimulationNPI(data=data, n_samples=10,
                                   epi_model="rost_model")

    simulation.generate_lhs()
    simulation.get_analysis_results()
    # simulation.calculate_prcc_values()
    # simulation.plot_prcc_values()


if __name__ == '__main__':
    main()
