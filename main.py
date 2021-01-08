from simulation_npi import SimulationNPI
from plotter import generate_stacked_plots


def main():
    simulation = SimulationNPI()
    simulation.run()


if __name__ == '__main__':
    # generate_stacked_plots()
    main()
