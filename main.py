from src.simulation_npi import SimulationNPI
from src.data_transformer import Transformer


def main():

    data = Transformer()

    simulation = SimulationNPI(sim_state=data.sim_state, sim_obj=data.sim_obj)
    simulation.generate_lhs()
    simulation.prcc_plots_generation()
    simulation.get_analysis_results()


if __name__ == '__main__':

    main()
