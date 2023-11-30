import json
import os

import numpy as np
import xlrd


PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))


class DataLoader:
    def __init__(self):
        self._model_parameters_data_file = os.path.join(PROJECT_PATH, "../data", "model_parameters.json")
        self._uk_model_parameters_data_file = os.path.join(PROJECT_PATH, "../data", "uk_model_parameters.json")

        self._contact_data_file = os.path.join(PROJECT_PATH, "../data", "contact_matrices.xls")
        self._uk_contact_data_file = os.path.join(PROJECT_PATH, "../data", "uk_contact_matrices.xls")

        self._age_data_file = os.path.join(PROJECT_PATH, "../data", "age_distribution.xls")
        self._uk_age_data_file = os.path.join(PROJECT_PATH, "../data", "uk_age_distribution.xls")

        self._get_age_data()
        self._get_model_parameters_data()
        self._get_uk_model_parameters_data()
        self._get_contact_mtx()
        self._get_uk_contact_mtx()

    def _get_age_data(self):
        files = [self._age_data_file, self._uk_age_data_file]
        for file in files:
            wb = xlrd.open_workbook(file)
            sheet = wb.sheet_by_index(0)
            datalist = np.array([sheet.row_values(i) for i in range(0, sheet.nrows)])
            wb.unload_sheet(0)
            if file == self._age_data_file:
                self.age_data = datalist
            else:
                self.uk_age_data = datalist

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

    def _get_uk_model_parameters_data(self):
        with open(self._uk_model_parameters_data_file) as f:
            parameters = json.load(f)
            self.uk_model_parameters_data = dict()
            for param in parameters.keys():
                param_value = parameters[param]["value"]
                if isinstance(param_value, list):
                    self.uk_model_parameters_data.update({param: np.array(param_value)})
                else:
                    self.uk_model_parameters_data.update({param: param_value})

    def _get_contact_mtx(self):
        wb = xlrd.open_workbook(self._contact_data_file)
        contact_matrices = dict()
        for idx in range(4):
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

    def _get_uk_contact_mtx(self):
        wb = xlrd.open_workbook(self._uk_contact_data_file)
        uk_contact_matrices = dict()
        for idx in range(4):
            sheet = wb.sheet_by_index(idx)
            datalist = np.array([sheet.row_values(i) for i in range(0, sheet.nrows)])
            cm_type = wb.sheet_names()[idx]
            wb.unload_sheet(0)
            datalist = self.transform_uk_matrix(datalist)
            uk_contact_matrices.update({cm_type: datalist})
        self.uk_contact_data = uk_contact_matrices

    def transform_uk_matrix(self, matrix: np.ndarray):
        # Get uk age vector as a column vector
        uk_age_distribution = self.uk_age_data.reshape((-1, 1))  # (16, 1)

        # Get uk matrix of total number of contacts
        uk_matrix = matrix * uk_age_distribution

        # Get symmetric matrix
        output = (uk_matrix + uk_matrix.T) / 2
        # Get contact matrix
        output /= uk_age_distribution  # divides and assign the result to output    (16, 16)
        return output


