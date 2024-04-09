import src
from src.dataloader import DataLoader


def main():
    data = DataLoader(country="hungary")

    simulation = src.SimulationNPI(data=data, n_samples=10,
                                   epi_model="rost", country="hungary")
    do_generate_lhs = False
    do_calculate_prcc = True
    do_generate_analysis = False
    do_plot_contact_manipulation = False

    if do_generate_lhs:
        simulation.generate_lhs()
    if do_calculate_prcc:
        simulation.calculate_prcc_values()
        simulation.plot_prcc_values()
    if do_generate_analysis:
        simulation.generate_analysis_results()
    if do_plot_contact_manipulation:
        simulation.plot_max_values_contact_manipulation()


if __name__ == '__main__':
    main()
