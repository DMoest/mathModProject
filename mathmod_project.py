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
from datetime import datetime
from pprint import pprint as pp

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import csv


def parse_csv_data(data_file_path, key, delimiter=','):
    """
    Parses a csv file.

    :param data_file_path: Path to the csv file
    :type data_file_path: str
    :param key: Key for the parsed data
    :type key: str
    :param delimiter: Delimiter for the csv file
    :type delimiter: str
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
    :type data_frame: pd.DataFrame

    :return: Pandas DataFrame
    :rtype: pd.DataFrame
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
        data_frame_name = f'{data}'
        globals()[data_frame_name] = pd.read_csv(parsed_data_paths[data], delimiter=',')
        globals()[data_frame_name] = globals()[data_frame_name].rename(columns={'Close': f'{data}'})
        input_data_paths[data] = globals()[data_frame_name]

        print(f'Cleaning up data and create a Pandas data frame for {data}... \n')
        globals()[data_frame_name] = clean_up_csv_data(globals()[data_frame_name])
        data_frames[data] = pd.DataFrame(globals()[data_frame_name])

    print("Parsed data paths: \n", json.dumps(parsed_data_paths, indent=4), "\n")
    print('Done parsing files, cleaning up CSV data files and creating Pandas data frames. \n')
    pp(data_frames.values())


def concatenate_data_frames(input_data_frames):
    """
    The function concatenates all data frames into one and drops all rows with NaN values.

    :param input_data_frames: A list of data frames to be concatinated
    :type input_data_frames: dict

    :return A data frame with all the data from the input data frames concatinated into one data frame.
    :rtype pd.DataFrame
    """
    concatenated_data_frame = pd.concat(input_data_frames, axis=1).dropna()

    return concatenated_data_frame


def plot_concatinated_data_frame(input_data_frame, show=False):
    """
    This function takes in a data frame and plots it.

    :param input_data_frame: The data frame that you want to plot.
    :type input_data_frame: pd.DataFrame
    :param show: If True, the plot will be shown
    :type bool
    :type input_data_frame: pd.DataFrame

    :return: None
    """
    input_data_frame.plot()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.subplots_adjust(right=0.75, bottom=0.2)
    plt.savefig('plots/value_over_time/all_data_frames_concatenated.png')

    if show:
        plt.show()


def lineplot_data_frames(input_data_frames, show=False):
    """
    Plots the data from the Pandas data frames.

    :param input_data_frames: A dictionary of Pandas data frames
    :type input_data_frames: dict
    :param show: If True, the plot will be shown
    :type show: bool

    :return: None
    """
    print('Staring to create Seaborn lineplots from data frames... ')

    for data_frame in input_data_frames:
        print(f'Plotting value over time data for {data_frame}... ')
        input_data_frames[data_frame].plot()
        sns.lineplot(data=input_data_frames[data_frame], x='Date', y=data_frame).figure.savefig(
            f'./plots/value_over_time/value_over_time_{data_frame}.png', dpi=300)

        if show:
            plt.show()

    print('Done plotting value over time data. \n')


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

        this_df = input_data_frames[data_frame][f'{data_frame}']

        # Create a Pandas data series with the descriptive statistics
        output_data_frame = pd.Series({'Count': int(this_df.count()),
                                       'Min': this_df.min(),
                                       'Max': this_df.max(),
                                       'Mean': this_df.mean(),
                                       'Median': this_df.median(),
                                       'Std': this_df.std(),
                                       'Var': this_df.var(),
                                       }, index=['Count', 'Min', 'Max', 'Mean', 'Median', 'Std', 'Var'])

        # Output the descriptive statistics to a csv file
        output_data_frame.to_csv(f'./tables/individual_statistics/statistics_{data_frame}.csv', header=False)

        # Add data series to dictionary
        statistics[data_frame] = output_data_frame

    print('Done calculating descriptive statistics. \n')

    return statistics


def plot_descriptive_statistics_table(input_data_frame, show=False):
    """
    Plots the descriptive statistics table.

    :param input_data_frame: A Pandas data frame with the descriptive statistics.
    :type input_data_frame: pd.DataFrame.
    :param show: If True, the plot will be shown
    :type show: bool

    :return: None
    """
    table = plt.table(
        cellText=input_data_frame.values.round(3),
        colLabels=input_data_frame.columns,
        rowLabels=input_data_frame.index,
        cellLoc='left',
        loc='center',
    )
    plt.axis('off')
    table.auto_set_font_size(False)
    table.set_fontsize(4)
    plt.tight_layout()
    plt.savefig(
        './tables/summarized_descriptive_statistics.png',
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

    if show:
        plt.show()


def calculate_correlation_matrix_and_plot_heatmap(input_data_frame, show=False):
    """
    This function takes in a data frame, creates a correlation matrix, and plots a heatmap of the correlation matrix.

    :param input_data_frame: The data frame that contains the data that you want to calculate the correlation matrix for
    :type input_data_frame: pd.DataFrame
    :param show: If True, the plot will be shown
    :type show: bool

    :return: Correlation matrix
    :rtype: pd.DataFrame
    """
    # Create Correlation matrix
    correlation_matrix = input_data_frame.corr()

    # Plot heatmap of correlation matrix
    correlation_heatmap = sns.heatmap(correlation_matrix, annot=True, annot_kws={'size': 4}, cmap="flare")
    correlation_heatmap.xaxis.set_tick_params(labelsize=5)
    correlation_heatmap.yaxis.set_tick_params(labelsize=5)
    plt.title('Correlations', fontsize=20)
    plt.xlabel('Cryptocurrencies', fontsize=10)
    plt.ylabel('Cryptocurrencies', fontsize=10)
    plt.text(1, 1, '', ha='center', va='center', fontsize=4)
    plt.xticks(rotation=75)
    plt.yticks(rotation=15)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Save figure to file
    correlation_heatmap.figure.savefig('./plots/heatmaps/correlation_heatmap.png', bbox_inches='tight', dpi=300)

    # Show plots
    if show:
        plt.show()

    return correlation_matrix


# Normal distribution plots
def plot_normal_distributions(input_data_frames, show=False):
    """
    This function plots the normal distribution of each of the input data frames

    :param input_data_frames: A dictionary of data frames
    :type input_data_frames: dict
    :param show: If True, the plot will be shown, defaults to False (optional)
    :type show: bool, optional
    """
    print("Starting to plot normal distributions...")

    for data in input_data_frames:
        print(f"Plotting the normal distribution of {data}...")
        sns.displot(
            data=input_data_frames[data][f'{data}'],
            x=input_data_frames[data][f'{data}'].values,
            kde=True,
            label=data,
            stat="probability",
            fill=True,
        )
        plt.title(f'Normal distribution of {data}', fontsize=20)
        plt.savefig(f'./plots/normal_distribution/normal_distribution_{data}.png', bbox_inches='tight', dpi=300)

        if show:
            plt.show()

    print("Done plotting normal distributions. \n")


def calculate_simple_linear_regressions(input_data_frames, show=False):
    """
    > This function takes in a dictionary of data frames, and returns a dictionary of linear regression models

    :param input_data_frames: A dictionary of data frames
    :type input_data_frames: dict
    :param show: If True, the plot will be shown. If False, the plot will be saved to a file, defaults to False (optional)
    :type show: bool, optional
    """
    print("Starting to calculate simple linear regressions...")

    for data_frame in input_data_frames:
        print(f"Calculating simple linear regression for {data_frame}...")

        # Fit the linear regression model
        x = input_data_frames[data_frame].index.map(datetime.toordinal).values.reshape(-1, 1)
        y = input_data_frames[data_frame][data_frame].values.reshape(-1, 1)

        # Create a linear regression object
        linear_regression = LinearRegression()
        model = linear_regression.fit(x, y)

        # Get the slope and intercept of the line best fit
        r_squared = model.score(x, y)

        # # Get the slope and intercept of the line best fit
        slope = linear_regression.coef_[0][0]
        intercept = linear_regression.intercept_[0]
        y_predict = model.predict(x)
        fx = intercept + slope * x

        # Store the linear regression model in a dictionary
        linear_regressions[data_frame] = {
            'x': x,
            'y': y,
            'y_predict': y_predict,
            'fx': fx,
            'r_squared': r_squared,
            'slope': slope,
            'intercept': intercept
        }

        # Plot the linear regression
        sns.set_style('darkgrid')
        # plot = sns.regplot(x, y_predict, ci=95, scatter_kws={'color': 'orange', 's': 2},
        #                    line_kws={'color': 'red', 'lw': 1})
        dot_size = 0.7
        plot = sns.regplot(x, y, ci=95, scatter_kws={'color': 'blue', 's': 2}, line_kws={'color': 'red', 'lw': 0.5},
                           dropna=True, marker='.')

        # plt.scatter(x, y, s=dot_size, color='blue')
        plt.title(f'{data_frame} over Time', fontsize=16)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Value', fontsize=10)
        plot.plot(x, y_predict, color='red', linewidth=0.5)

        plt.savefig(f'./plots/linear_regression/linear_regression_{data_frame}.png', bbox_inches='tight',
                    dpi=300)

        if show:
            plt.show()

    print('Done calculating simple linear regressions. \n')


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

# Dictionary for the data frames
parsed_data_paths = {}  # Dictionary for parsed data paths
data_frames = {}  # Dictionary for Pandas data frames
linear_regressions = {}  # Dictionary for linear regression models

# Clean up data and create Pandas data frames
clean_all_data_frames(data_paths)

# Concatenate all data frames into one
concat_data_frame = concatenate_data_frames(data_frames)

# Calculate descriptive statistics from the data frames and concatenate them into one data frame for plotting
descriptive_statistics = calculate_description_statistics(data_frames)
descriptive_statistics_table = concatenate_data_frames(descriptive_statistics)

# TODO: Fix plotting, uncomment before handing in assignment.
# Run plot data functions
plot_concatinated_data_frame(concat_data_frame, True)
lineplot_data_frames(data_frames, True)
plot_descriptive_statistics_table(descriptive_statistics_table, True)
calculate_correlation_matrix_and_plot_heatmap(concat_data_frame, True)
plot_normal_distributions(data_frames, True)
calculate_simple_linear_regressions(data_frames, True)
