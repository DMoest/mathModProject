#!/usr/bin/env python3

"""
This is the final and examining project of the course "Mathematical Modeling" at Blekinge Institute of Technology.

@date:      2022-08-29
@student:   Daniel Andersson
@akronym:   daap19
@course:    Mathematical Modeling, MA1487
@teacher:   Simon Nilsson
"""

import json
from pprint import pprint as pp

import pandas as pd

import csv


def parse_csv_data(data_file_path, key, delimiter=','):
    """
    Parses a csv file.

    :param data_file_path:
    :param delimiter:
    :return:
    """

    # Remove 4 last letters (.csv) from path name and add '-parsed.csv'
    parsed_file_name = data_file_path[:-4] + '-parsed.csv'
    with open(data_file_path, 'r') as in_file, open(parsed_file_name, 'w') as out_file:
        parsed_header = False
        reader = csv.reader(in_file, delimiter=delimiter)
        write = csv.writer(out_file, delimiter=delimiter)

        for i, row in enumerate(reader):
            if len(row) > 0 and row[0] == 'SNo':
                parsed_header = True
            if parsed_header:
                write.writerow(row)

        parsed_data_paths[key] = parsed_file_name


def clean_up_csv_data(data_frame):
    """
    Cleans up a csv file. Removes unnecessary columns.

    :param data_frame: Pandas DataFrame
    :return: Pandas DataFrame
    """
    data_frame['Date'] = pd.to_datetime(data_frame['Date'], format='%Y-%m-%d %H:%M:%S')
    data_frame.drop(columns=['SNo', 'Symbol', 'High', 'Low', 'Volume'], inplace=True)
    data_frame.set_index('Date', inplace=True)

    return data_frame


def clean_all_data_frames(input_data_paths):
    """
    This function takes in a dictionary of input data paths, parses the data from the input files, cleans up the data, and
    creates Pandas data frames for each data set.

    The function returns a dictionary of output data paths.

    The function also creates a dictionary of Pandas data frames.

    :param input_data_paths: A dictionary of input data paths
    :type input_data_paths: dict

    :return: None
    """
    print("Input data paths: \n", json.dumps(input_data_paths, indent=4), "\n")

    for data in input_data_paths:
        # Parse data from input files
        print(f'Parsing CSV file for {data} data... ')
        parse_csv_data(input_data_paths[data], data)

        # Read data from parsed csv file
        print(f'Reading parsed CSV file for {data} data... ')
        data_frame_name = f'data_frame_{data}'
        globals()[data_frame_name] = pd.read_csv(parsed_data_paths[data], delimiter=',')
        input_data_paths[data] = globals()[data_frame_name]

        print(f'Cleaning up data and create a Pandas data frame for {data}... \n')
        globals()[data_frame_name] = clean_up_csv_data(globals()[data_frame_name])
        data_frames[data] = pd.DataFrame(globals()[data_frame_name])

    print("Parsed data paths: \n", json.dumps(parsed_data_paths, indent=4), "\n")
    print('Done parsing files, cleaning up CSV data files and creating Pandas data frames. \n')
    pp(f'Data frames created: \n{data_frames}\n ')


# Paths for the csv files containing the data
data_paths = {
    'Bitcoin': './csv/coin_Bitcoin.csv',
    'Ethereum': './csv/coin_Ethereum.csv',
    'Monero': './csv/coin_Monero.csv',
    'Dogecoin': './csv/coin_Dogecoin.csv',
    'Polkadot': './csv/coin_Polkadot.csv',
}

# Paths for parsed csv files
parsed_data_paths = {}
data_frames = {}
clean_all_data_frames(data_paths)

# Dictionary for the dataframes

#     # Clear data from parsed csv file
#     dynamic_cleared_dataframe_name = f'cleared_data_frame_{data}'
#     globals()[dynamic_cleared_dataframe_name] = clear_smhi_data_pandas(globals()[dynamic_dataframe_name])
#     data_frames[data] = globals()[dynamic_cleared_dataframe_name]
#     # data_frames[data] = data_frames[data].rename(columns={'Lufttemperatur': f'LT_{data[:3]}'})
#
#     # Plot data
#     data_frames[data].plot()
#
# # Concatenate dataframes & drop NaN values. This is done to get a dataframe with all the data.
# concat_data_frame = pd.concat(data_frames, axis=1).dropna()
# print("Concatenated data frame:")
# print(concat_data_frame)
#
# concat_data_frame.plot(title='Temperatur västerås, arlanda, lanvetter, sturup, kiruna', ylabel='Temperatur',
#                        xlabel='Datum')
#
# X = concat_data_frame.index.map(datetime.toordinal).values.reshape(-1, 1)
# Y = concat_data_frame.iloc[:, 1].values.reshape(-1, 1)
#
# print(f'X: {X}')
# print(f'Y: {Y}')
#
# # Create linear regression model
# linear_reg = LinearRegression()
# linear_reg.fit(X, Y)
# Y_pred = linear_reg.predict(X)
# print(f'Y_pred: {Y_pred}')
# Y_residual = Y - Y_pred
# print(f'Y_residual: {Y_residual}')
#
# # Plot linear regression model
# plt.figure()
# plt.scatter(X, Y)
# plt.plot(X, Y_pred, color='red')
#
# residual = concat_data_frame['västerås'].values - Y_pred
# concat_data_frame['residual_väs'] = residual
# concat_data_frame['residual_väs'].plot()
#
# residual_variance = residual.var()
# correlation = concat_data_frame.corr()
#
# # Seaborn plot
# sns.set_theme(style="darkgrid")
# sns.distplot(concat_data_frame, kde=False)
# print(f"{concat_data_frame.columns[0][1]}")
#
# # sns.heatmap(correlation)
# # plt.title('Correlation between sites')
#
# plt.show()
#
# print(correlation)
#
# # residuals = {}
# #
# # for data in concat_data_frame:
# #     dynamic_residual_name = f'residual_{data}'
# #     globals()[dynamic_residual_name] = data_frames[data].values - Y_pred.squeeze()
# #     data_frames[dynamic_residual_name] = globals()[dynamic_residual_name]
# #     data_frames[dynamic_residual_name].plot()
