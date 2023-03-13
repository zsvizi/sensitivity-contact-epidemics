from src.simulation_npi import SimulationNPI
from src.data_transformer import DataTransformer


def main():

    data_tr = DataTransformer()
    simulation = SimulationNPI(data_tr=data_tr, sim_state=data_tr.sim_state_data)
    simulation.generate_prcc_values()


if __name__ == '__main__':
    main()
