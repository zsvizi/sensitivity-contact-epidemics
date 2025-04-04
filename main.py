import src
from src.dataloader import DataLoader


def main():
    data = DataLoader(country="Hungary_maszk")
    simulation = src.SimulationNPI(data=data,
                                   n_samples=10000,
                                   epi_model="rost_maszk",
                                   country="Hungary_maszk",
                                   strategy="baseline",
                                   is_kappa_applied=False)
    simulation.generate_lhs()
    simulation.calculate_prcc_values()
    simulation.plot_prcc_values()
    # simulation.generate_analysis_results()
    # simulation.plot_max_values_contact_manipulation()


if __name__ == '__main__':
    main()


