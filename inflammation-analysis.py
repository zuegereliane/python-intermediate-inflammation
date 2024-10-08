#!/usr/bin/env python3
"""Software for managing and analysing patients' inflammation data in our imaginary hospital."""

import argparse
import os

from inflammation import models, views
from inflammation.compute_data import CSVDataSource, JSONDataSource, analyse_data


def main(args):
    """The MVC Controller of the patient inflammation data system.

    The Controller is responsible for:
    - selecting the necessary models and views for the current task
    - passing data between models and views
    """
    in_files = args.infiles
    if not isinstance(in_files, list):
        in_files = [args.infiles]

    if args.full_data_analysis:
        _, extension = os.path.splitext(in_files[0])
        if extension == '.json':
            data_source = JSONDataSource(os.path.dirname(in_files[0]))
        elif extension == '.csv':
            data_source = CSVDataSource(os.path.dirname(in_files[0]))
        else:
            raise ValueError(f'Unsupported data file format: {extension}')

        daily_standard_deviation = analyse_data(data_source)
        graph_data = {
            'standard deviation by day': daily_standard_deviation,
        }
        views.visualize(graph_data)
        return

    for filename in in_files:
        inflammation_data = models.load_csv(filename)

        view_data = {'average': models.daily_mean(inflammation_data),
                     'max': models.daily_max(inflammation_data),
                     'min': models.daily_min(inflammation_data)
                     }

        views.visualize(view_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='A basic patient inflammation data management system')

    parser.add_argument(
        'infiles',
        nargs='+',
        help='Input CSV(s) containing inflammation series for each patient')

    parser.add_argument(
        '--full-data-analysis',
        action='store_true',
        dest='full_data_analysis')

    args = parser.parse_args()

    main(args)
