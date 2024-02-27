import numpy as np
import ast
from time import sleep
from tqdm import tqdm  # Import tqdm for progress bar
import csv
import os

from src.seirsv.seirsv_simulation_base import SimulationBase
from src.seirsv.dataloader import seirSVDataLoader
from src.seirsv.r0_seirsv import R0SeirSVModel
from src.seirsv.plot_tornado import Tornado_plot


class SimulationSeirSV(SimulationBase):
    def __init__(self, data: seirSVDataLoader, n_samples: int = 1,
                 all_contact_type: str = "all",  phy_contact_type: str = "physical",
                 model_seasonality: str = "Yes", no_model_seasonality: str = "Yes",
                 model_seed: str = "Yes",
                 varying_vaccination_rate: str = "Yes",
                 vaccine_induced_halved: str = "Yes", same_vaccination: str = "Yes") -> None:

        # set attributes
        self.varying_vaccination_rate = varying_vaccination_rate
        self.vaccine_induced_halved = vaccine_induced_halved
        self.same_vaccination = same_vaccination
        self.model_seasonality = model_seasonality
        self.no_model_seasonality = no_model_seasonality
        self.model_seed = model_seed
        self.all_contact_type = all_contact_type
        self.phy_contact_type = phy_contact_type

        super().__init__(data=data)
        self.n_samples = n_samples

        # defined parameters
        self.r0_choices = [1.1, 1.4, 1.8, 2.2, 4.35]
        self.same_vac_param = np.full(15, 0.5)  # 50% vaccination rate for all age groups (n=15)
        self.psi_i = [0.8, 0.8, 0.8, 0.8, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
                      0.75, 0.75, 0.5, 0.5]  # 80% vaccination rate of younger age groups
        self.omega_v = 0.8333333   # mean duration of vaccine induced, halved

        # create empty list to store results
        self.all_cont = np.empty((len(self.r0_choices), 1))  # Initialize a NumPy array
        self.seed = np.empty((len(self.r0_choices), 1))
        self.no_seasonality = np.empty((len(self.r0_choices), 1))
        self.vaccine_induced = np.empty((len(self.r0_choices), 1))
        self.no_seed = np.empty((len(self.r0_choices), 1))
        self.physical_cont = np.empty((len(self.r0_choices), 1))
        self.physical18 = np.empty((len(self.r0_choices), 1))
        self.vac_same_people = np.empty((len(self.r0_choices), 1))
        self.vac_spread = np.empty((len(self.r0_choices), 1))

    def base_case_scenario(self):
        """
        Assumes base_r0 = 1.8, 80 percent vaccination of the age groups 0-19, all_contact, seasonality,
        no seeding, vaccine induced not halved, not same vaccination
        :return: scalar
        """
        base_case_results = []
        r0generator = R0SeirSVModel(param=self.params)
        r0 = r0generator.get_eig_val(
            contact_mtx=self.all_contact,
            susceptibles=self.susceptibles.reshape(1, -1),
            population=self.population
        )[0]
        beta = self.r0_choices[2] / r0
        self.params.update({"beta": beta})
        self.sim_state.update(
            {"base_r0": self.r0_choices[2],
             "beta": beta,
             "params": self.psi_i,
             "r0generator": r0generator})

        # calculate incidence cases, r0=1.8, 80% vaccination, with seasonality and no vaccine vary
        t = np.arange(0, 11 * 50, 1)  # run simulations from 2009 to 2020
        sol = self.model.get_solution(
            init_values=self.model.get_initial_values, t=t, parameters=self.params,
            cm=self.all_contact)
        base_case_results.append(self.model.seasonality_incident_cases)
        return base_case_results

    def different_scenarios(self):
        """
        Assumes different base_r0, contact, seasonality, vaccine  of the age groups 0-19
        :return: sensitivity results
        """
        results = {
            'base_case': [],
            'all_cont': [],
            'physical_cont': [],
            'no_seasonality': [],
            'seed': [],
            'vaccine_induced': [],
            'same_vaccine': [],
            'spread_vaccine': []
        }

        for idx, base_r0 in tqdm(enumerate(self.r0_choices),
                                 desc="Simulating scenarios",
                                 total=5):
            sleep(0.2)
            r0generator = R0SeirSVModel(param=self.params)
            r0 = r0generator.get_eig_val(
                contact_mtx=self.all_contact,
                susceptibles=self.susceptibles.reshape(1, -1),
                population=self.population
            )[0]
            beta = base_r0 / r0
            self.params.update({"beta": beta})
            self.sim_state.update(
                {"base_r0": self.r0_choices,
                 "beta": beta,
                 "r0generator": r0generator})
            print("Simulations for base_r0=" + str(base_r0))

            t = np.arange(0, 12 * 365, 1)  # run simulations from 2009 to 2020
            if self.all_contact_type == "all":
                sol = self.model.get_solution(
                    init_values=self.model.get_initial_values, t=t,
                    parameters=self.params,
                    cm=self.all_contact)
                self.all_cont[idx, :] = self.model.seasonality_incident_cases  # we need for: r0=1.4, 1.8, 2.2, 4.35
                results['all_cont'].append(self.all_cont[idx, :])
            else:
                results['all_cont'].append(None)
            if self.phy_contact_type == "physical":
                sol = self.model.get_solution(
                    init_values=self.model.get_initial_values, t=t,
                    parameters=self.params,
                    cm=self.physical_contact)
                self.physical_cont[idx, :] = self.model.seasonality_incident_cases
                results['physical_cont'].append(self.physical_cont[idx, :])  # we need: r0=1.1, 1.8
            else:
                results['physical_cont'].append(None)
            if self.no_model_seasonality == "Yes":
                self.params["z"] = 0
                sol = self.model.get_solution(
                    init_values=self.model.get_initial_values, t=t,
                    parameters=self.params,
                    cm=self.all_contact)
                self.no_seasonality[idx, :] = self.model.seasonality_incident_cases
                results['no_seasonality'].append(self.no_seasonality[idx, :])
            else:
                results['no_seasonality'].append(None)
            if self.model_seed == "Yes":
                self.params['param'] = self.params['seeding']
                sol = self.model.get_solution(
                    init_values=self.model.get_initial_values, t=t,
                    parameters=self.params,
                    cm=self.all_contact)
                self.seed[idx, :] = self.model.incident_cases_seed
                results['seed'].append(self.seed[idx, :])  # uses base line, r0=1.8
            else:
                results['seed'].append(None)

            if self.vaccine_induced_halved == "Yes":
                self.params["omega_v"] = self.omega_v
                sol = self.model.get_solution(
                    init_values=self.model.get_initial_values, t=t,
                    parameters=self.params,
                    cm=self.all_contact)
                self.vaccine_induced[idx, :] = self.model.vaccine_induced
                results['vaccine_induced'].append(self.vaccine_induced[idx, :])  # r0=1.8
            else:
                results['vaccine_induced'].append(None)
            if self.same_vaccination == "Yes":
                self.params["psi_i"] = self.same_vac_param
                sol = self.model.get_solution(
                    init_values=self.model.get_initial_values, t=t,
                    parameters=self.params,
                    cm=self.all_contact)
                self.vac_same_people[idx, :] = self.model.same_vac
                results['same_vaccine'].append(self.vac_same_people[idx, :])  # r0=1.8
            else:
                results['same_vaccine'].append(None)
            if self.varying_vaccination_rate == "Yes":
                self.params["psi_i"] = self.params["vary_vaccination_rate"]
                sol = self.model.get_solution(
                    init_values=self.model.get_initial_values, t=t,
                    parameters=self.params,
                    cm=self.all_contact)
                self.vac_spread[idx, :] = self.model.incident_cases_vac
                results['spread_vaccine'].append(self.vac_spread[idx, :])  # r0=1.8
            else:
                results['spread_vaccine'].append(None)
        return results

    def save_sensitivity_results(self):
        results = self.different_scenarios()
        base_case = self.base_case_scenario()
        directory = './sens_data/'
        file_name = 'simulation_results.csv'
        file_path = os.path.join(directory, file_name)
        # Create or open the CSV file in write mode
        with open(file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write the header row with column names
            csv_writer.writerow(['Scenario', 'Result'])
            # Save results for base case
            csv_writer.writerow(['80% LAIV vaccination, 2 to 17 yr olds',
                                 base_case])
            # Save results for all_cont_r018
            csv_writer.writerow(['Homogeneous mixing (R0 1.8)',
                                 results['all_cont'][2] if
                                 results['all_cont'] else None])
            # Save results for vaccination_spread
            csv_writer.writerow(['Vaccination spread',
                                 results['spread_vaccine'][2] if
                                 results['spread_vaccine']
                else None])
            # Save results for vac_induced
            csv_writer.writerow(['Duration of vaccine protection halved',
                                 results['vaccine_induced'][2] if
                                 results['vaccine_induced'] else None])
            # Save results for same_vaccination
            csv_writer.writerow(['Vaccinate same people each year',
                                 results['same_vaccine'][2] if
                                 results['same_vaccine'] else None])
            # Save results for phy_cont18
            csv_writer.writerow(['Physical contact matrix (R0=1.8)',
                                 results['physical_cont'][2] if
                                 results['physical_cont'] else None])

            # Save results for infection_seed
            csv_writer.writerow(['Seed value 10',
                                 results['seed'][2] if
                                 results['seed'] else None])
            csv_writer.writerow(['R0 2.2',
                                 results['all_cont'][3] if
                                 results['all_cont'] else None])
            # Save results for all_cont_r0435
            csv_writer.writerow(['Homogeneous mixing (R0 4.35)',
                                 results['all_cont'][4] if
                                 results['all_cont'] else None])
            # Save results for no_seasonality
            csv_writer.writerow(['No seasonality',
                                 results['no_seasonality'][2] if
                                 results['no_seasonality'] else None])

            csv_writer.writerow(['R0 1.4',
                                 results['all_cont'][1] if
                                 results['all_cont'] else None])

            # Save results for phy_cont11
            csv_writer.writerow(['Physical contact matrx (R0=1.1)',
                                 results['physical_cont'][0] if
                                 results['physical_cont'] else None])

    def plot_sensitivity_results(self):
        # read the saved files
        directory = './sens_data/'
        file_name = 'simulation_results.csv'
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            # Skip the header row
            header = next(csv_reader)
            # Create a dictionary to store results
            results_dict = {}
            # Iterate through rows
            for row in csv_reader:
                scenario, result = row
                results_dict[scenario] = result
            # Convert strings to list and extract numerical values
            result_values = {key: float(ast.literal_eval(value)[0]) for key,
                                                                        value in results_dict.items()}

            # reference is the base case scenario
            reference_scenario = '80% LAIV vaccination, 2 to 17 yr olds'
            reference_value = result_values[reference_scenario]
            # calculate the averted infections, z - z'
            deviations = {scenario: float(ast.literal_eval(result)[0]) -
                                    reference_value if result is not None else None
                          for scenario, result in results_dict.items()}

            # Remove the reference scenario from the deviations
            del deviations[reference_scenario]

            # plot the sensitivity results
            plot = Tornado_plot(sensitivity_results=result)
            plot.generate_tornado_plot(deviations=deviations,
                                       reference_value=reference_value,
                      reference_scenario=reference_scenario)
