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
import seaborn as sns
from matplotlib import pyplot as plt

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
    data_frame.drop(columns=['SNo', 'Symbol', 'High', 'Low', 'Open', 'Volume', 'Marketcap'], inplace=True)
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
        globals()[data_frame_name] = globals()[data_frame_name].rename(columns={'Close': f'Value {data}'})
        input_data_paths[data] = globals()[data_frame_name]

        print(f'Cleaning up data and create a Pandas data frame for {data}... \n')
        globals()[data_frame_name] = clean_up_csv_data(globals()[data_frame_name])
        data_frames[data] = pd.DataFrame(globals()[data_frame_name])

    print("Parsed data paths: \n", json.dumps(parsed_data_paths, indent=4), "\n")
    print('Done parsing files, cleaning up CSV data files and creating Pandas data frames. \n')
    pp(data_frames.values())


def lineplot_data_frames(input_data_frames):
    """
    Plots the data from the Pandas data frames.

    :param input_data_frames: A dictionary of Pandas data frames
    :type input_data_frames: dict

    :return: None
    """
    print(f'Staring to create Seaborn lineplots from data frames... ')
    for data_frame in input_data_frames:
        print(f'Plotting data from {data_frame}... ')
        input_data_frames[data_frame].plot()

        sns.lineplot(data=input_data_frames[data_frame], x='Date', y=f'Value {data_frame}').figure.savefig(
            f'./plots/plot_{data_frame}.png', dpi=300)

        plt.show()


def calculate_description_statistics(input_data_frames):
    """
    Calculates the description statistics for the Pandas data frames.

    :param input_data_frames: A dictionary of Pandas data frames
    :type input_data_frames: dict

    :return: None
    """
    statistics = {}

    for data_frame in input_data_frames:
        print(f'Calculating descriptive statistics for {data_frame}... ')
        # print(input_data_frames[data_frame][f'Value {data_frame}'].describe())

        df_values = input_data_frames[data_frame][f'Value {data_frame}']

        # Create a Pandas data series with the descriptive statistics
        data_series = pd.Series({'Count': df_values.count(),
                                 'Min': df_values.min(),
                                 'Max': df_values.max(),
                                 'Mean': df_values.mean(),
                                 'Median': df_values.median(),
                                 'Std': df_values.std(),
                                 'Var': df_values.var(),
                                 # 'Skew': df_values.skew(),
                                 # 'Kurt': df_values.kurt(),
                                 })

        # Output the descriptive statistics to a csv file
        data_series.to_csv(f'./statistics/statistics_{data_frame}.csv', header=False)

        # Add data series to dictionary
        statistics[data_frame] = data_series

    print('Done calculating descriptive statistics. \n')

    return statistics


# Paths for the csv files containing the data
data_paths = {
    'Aave': './csv/coin_Aave.csv',
    'BinanceCoin': './csv/coin_BinanceCoin.csv',
    'Bitcoin': './csv/coin_Bitcoin.csv',
    'Cardano': './csv/coin_Cardano.csv',
    'ChainLink': './csv/coin_ChainLink.csv',
    'Cosmos': './csv/coin_Cosmos.csv',
    'CryptocomCoin': './csv/coin_CryptocomCoin.csv',
    'Dogecoin': './csv/coin_Dogecoin.csv',
    'EOS': './csv/coin_EOS.csv',
    'Ethereum': './csv/coin_Ethereum.csv',
    'Iota': './csv/coin_Iota.csv',
    'Litecoin': './csv/coin_Litecoin.csv',
    'Monero': './csv/coin_Monero.csv',
    'NEM': './csv/coin_NEM.csv',
    'Polkadot': './csv/coin_Polkadot.csv',
    'Solana': './csv/coin_Solana.csv',
    'Stellar': './csv/coin_Stellar.csv',
    'Tether': './csv/coin_Tether.csv',
    'Tron': './csv/coin_Tron.csv',
    'Uniswap': './csv/coin_Uniswap.csv',
    'USDCoin': './csv/coin_USDCoin.csv',
    'XRP': './csv/coin_XRP.csv',
}

parsed_data_paths = {}  # Dictionary for parsed data paths
data_frames = {}  # Dictionary for Pandas data frames
linear_regression_data = {}  # Dictionary for linear regression data

# Clean up data and create Pandas data frames
clean_all_data_frames(data_paths)

# Concatinate all data frames into one and drop all rows with NaN values
concat_data_frame = pd.concat(data_frames, axis=1).dropna()
concat_data_frame.plot()
plt.show()

# Plot individual data frames
# lineplot_data_frames(data_frames)

# Calculate descriptive statistics
descriptive_statistics = calculate_description_statistics(data_frames)

descriptive_statistics_table = pd.concat(descriptive_statistics, axis=1).dropna()
table = plt.table(
    cellText=descriptive_statistics_table.values.round(4),
    colLabels=descriptive_statistics_table.columns,
    rowLabels=descriptive_statistics_table.index,
    cellLoc='left',
    loc='center',
)
plt.axis('off')
table.auto_set_font_size(False)
table.set_fontsize(4)
plt.tight_layout()
plt.savefig(
    './tables/descriptive_statistics.png',
    facecolor='w',
    edgecolor='w',
    format=None,
    bbox_inches=None,
    orientation='landscape',
    pad_inches=0.35,
    dpi=300
)
# make space for the table:
plt.subplots_adjust(left=0.05, bottom=0.05)
# plt.xticks([])
plt.show()  # Show plots

# Correlation matrix and heatmap
correlation_matrix = concat_data_frame.corr()
correlation_heatmap = sns.heatmap(correlation_matrix.round(2), annot=True, annot_kws={'size': 4}, cmap="flare")
correlation_heatmap.xaxis.set_tick_params(labelsize=5)
correlation_heatmap.yaxis.set_tick_params(labelsize=5)
plt.title('Correlations', fontsize=20)
plt.xlabel('Cryptocurrencies', fontsize=10)
plt.ylabel('Cryptocurrencies', fontsize=10)
plt.text(0.75, 0.75, ' ', ha='center', va='center', rotation=0, fontsize=4)
plt.subplots_adjust(left=0.3, bottom=0.35)
correlation_heatmap.figure.savefig('./heatmaps/correlation_heatmap.png', bbox_inches='tight', dpi=300)

# Show plots
plt.show()
