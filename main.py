from simulation_npi import SimulationNPI
from plotter import generate_stacked_plots, plot_contact_matrix_as_grouped_bars, plot_2d_contact_matrices


def main():
    # plot_2d_contact_matrices()
    simulation = SimulationNPI()
    simulation.run()


if __name__ == '__main__':

    main()
    # generate_stacked_plots()
    # plot_contact_matrix_as_grouped_bars()
