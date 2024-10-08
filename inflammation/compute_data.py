"""Module containing mechanism for calculating standard deviation between datasets.
"""

import glob
import os
import json
import numpy as np

from inflammation import models, views

class CSVDataSource:
    """Loads all the inflammation CSV files within a specified directory.
    data_dir: input directory containing CSV files
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        """Load a numpy array from a CSV file
        """
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data CSV files found in path {self.data_dir}")
        data = map(models.load_csv, data_file_paths)
        return list(data)

class JSONDataSource:
    """Loads all the inflammation JSON files within a specified directory.
       data_dir: input directory containing JSON files
       """
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):

        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.json'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data JSON files found in path {self.data_dir}")
        data = map(models.load_json, data_file_paths)
        return list(data)

def analyse_data(data_source):
    """Calculates the standard deviation by day between datasets.

    Gets all the inflammation data from CSV files within a directory,
    works out the mean inflammation value for each day across all datasets,
    then plots the graphs of standard deviation of these means.

    data_source: instance of Class CSVDataSource

    """
    data = data_source.load_inflammation_data()

    means_by_day = map(models.daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))

    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)

    return daily_standard_deviation
    # graph_data = {
    #     'standard deviation by day': daily_standard_deviation,
    # }
    # views.visualize(graph_data)
