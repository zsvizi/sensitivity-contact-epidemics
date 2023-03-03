from src.simulation_npi import SimulationNPI
from src.data_transformer import Transformer


def main():

    data_tr = Transformer()

    simulation = SimulationNPI(sim_state=data_tr.sim_state_data, sim_obj=data_tr)

    simulation.generate_lhs()
    simulation.prcc_plots_generation()


if __name__ == '__main__':
    main()
