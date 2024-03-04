import json
import os

import numpy as np
import xlrd


PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))


class DataLoader:
    def __init__(self, country: str):
        self.country = country

        if country == "Hungary":   # our model of analysis
            self._model_parameters_data_file = os.path.join(PROJECT_PATH, "../data", "model_parameters.json")
            self._contact_data_file = os.path.join(PROJECT_PATH, "../data", "contact_matrices.xls")
            self._age_data_file = os.path.join(PROJECT_PATH, "../data", "age_distribution.xls")
        if country == "usa":  # Modeling strict age-targeted mitigation strategies for COVID-19
            self._model_parameters_data_file = os.path.join(PROJECT_PATH, "../data", "usa_model_parameters.json")
            self._contact_data_file = os.path.join(PROJECT_PATH, "../data", "usa_matrices.xls")
            self._age_data_file = os.path.join(PROJECT_PATH, "../data", "usa_age_pop.xls")
            self._epidemic_data_file = os.path.join(PROJECT_PATH, "../data", "Epidemic_size_file.xls")
        elif country == "united_states":  # Projecting hospital utilization during the COVID-19 outbreaks in the US
            self._model_parameters_data_file = os.path.join(PROJECT_PATH, "../data", "US_model_parameters.json")
            self._contact_data_file = os.path.join(PROJECT_PATH, "../data", "US_model_contact.xls")
            self._age_data_file = os.path.join(PROJECT_PATH, "../data", "US_age_pop.xls")

        elif country == "UK":  # Influenza seir model in the UK
            self._model_parameters_data_file = os.path.join(PROJECT_PATH, "../data", "uk_model_parameters.json")
            self._contact_data_file = os.path.join(PROJECT_PATH, "../data", "uk_contact.xls")
            self._age_data_file = os.path.join(PROJECT_PATH, "../data", "uk_age_distribution.xls")

        # self._get_epidemic_data()
        self._get_age_data()
        self._get_model_parameters_data()
        self._get_contact_mtx()

    def _get_epidemic_data(self):
        wb = xlrd.open_workbook(self._epidemic_data_file)
        sheet = wb.sheet_by_index(0)
        datalist = np.array([sheet.row_values(i) for i in range(0, sheet.nrows)])
        wb.unload_sheet(0)
        self.epidemic_data = datalist

    def _get_age_data(self):
        wb = xlrd.open_workbook(self._age_data_file)
        sheet = wb.sheet_by_index(0)
        datalist = np.array([sheet.row_values(i) for i in range(0, sheet.nrows)])
        wb.unload_sheet(0)
        self.age_data = datalist

    def _get_model_parameters_data(self):
        # Load model parameters
        with open(self._model_parameters_data_file) as f:
            parameters = json.load(f)
        self.model_parameters_data = dict()
        for param in parameters.keys():
            param_value = parameters[param]["value"]
            if isinstance(param_value, list):
                self.model_parameters_data.update({param: np.array(param_value)})
            else:
                self.model_parameters_data.update({param: param_value})

    def _get_contact_mtx(self):
        wb = xlrd.open_workbook(self._contact_data_file)
        contact_matrices = dict()
        num_sheets = 2 if self.country in ["united_states", "UK"] else 4
        for idx in range(num_sheets):
            sheet = wb.sheet_by_index(idx)
            datalist = np.array([sheet.row_values(i) for i in range(0, sheet.nrows)])
            cm_type = wb.sheet_names()[idx]
            wb.unload_sheet(0)
            datalist = self.transform_matrix(datalist)
            contact_matrices.update({cm_type: datalist})
        self.contact_data = contact_matrices

    def transform_matrix(self, matrix: np.ndarray):
        # Get age vector as a column vector
        age_distribution = self.age_data.reshape((-1, 1))   # (16, 1)
        # Get matrix of total number of contacts
        matrix_1 = matrix * age_distribution        # (16, 16)
        # Get symmetrized matrix
        output = (matrix_1 + matrix_1.T) / 2
        # Get contact matrix
        output /= age_distribution   # divides and assign the result to output    (16, 16)
        return output
