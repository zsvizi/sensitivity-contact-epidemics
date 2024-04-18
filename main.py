import src
from src.dataloader import DataLoader


def main():
    data = DataLoader(country="Hungary")

    simulation = src.SimulationNPI(data=data, n_samples=10000,
                                   epi_model="rost", country="Hungary")
    simulation.generate_lhs()
    simulation.calculate_prcc_values(calculation_approach="mean")
    simulation.plot_prcc_values(calculation_approach="mean")
    simulation.generate_analysis_results()
    simulation.plot_max_values_contact_manipulation()


if __name__ == '__main__':
    main()
