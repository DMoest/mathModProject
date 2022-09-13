#!/usr/bin/env python3

"""
This is the final and examining project of the course
Mathematical Modeling at Blekinge Institute of Technology.

@date:      2022-08-29
@student:   Daniel Andersson
@akronym:   daap19
@course:    Mathematical Modeling, MA1487
@teacher:   Simon Nilsson
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn import metrics
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
    with open(data_file_path, 'r', encoding='utf-8') as in_file, open(parsed_file_name, 'w',
                                                                      encoding='utf-8') as out_file:
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
    data_frame.drop(
        ['SNo', 'Symbol', 'High', 'Low', 'Open', 'Volume', 'Marketcap'],
        axis=1,
        inplace=True,
    )
    data_frame.set_index(['Date'])

    return data_frame


def clean_all_data_frames(input_data_paths):
    """
    This function takes in a dictionary of input data paths,
    parses the data from the input files, cleans up the data, and
    creates Pandas data frames for each data set.

    The function returns a dictionary of output data paths.

    The function also creates a dictionary of Pandas data frames.

    :param input_data_paths: A dictionary of input data paths
    :type input_data_paths: dict

    :return: None
    """
    print("Input data paths: \n", json.dumps(input_data_paths, indent=4), "\n")

    for data in input_data_paths:
        print(f'Parsing CSV file for {data} data... ')
        parse_csv_data(input_data_paths[data], data)

        print(f'Read parsed CSV file for {data} data... ')
        data_frame_name = f'{data}'
        globals()[data_frame_name] = pd.read_csv(parsed_data_paths[data], delimiter=',')
        globals()[data_frame_name] = globals()[data_frame_name].rename(columns={'Close': 'Value'})
        input_data_paths[data] = globals()[data_frame_name]

        print(f'Cleaning up data and create a Pandas data frame for {data}... \n')
        globals()[data_frame_name] = clean_up_csv_data(globals()[data_frame_name])
        data_frames[data] = pd.DataFrame(globals()[data_frame_name])

    print(f" {len(parsed_data_paths)} parsed data paths: \n",
          json.dumps(parsed_data_paths, indent=4), "\n")
    print('Done parsing files, cleaning up CSV data files and creating Pandas data frames. \n')


def concatenate_data_frames(input_data_frames):
    """
    The function concatenates all data frames into one and drops all rows with NaN values.

    :param input_data_frames: A list of data frames to be concatenated
    :type input_data_frames: dict

    :return A data frame with all the data from the input data frames concatenated
    into one data frame.
    :rtype pd.DataFrame
    """
    print("Concatenating data frames... \n")
    concatenated_data_frame = pd.concat(input_data_frames, axis=1)

    return concatenated_data_frame


def plot_concatenated_data_frame(input_data_frame, show=False):
    """
    This function takes in a data frame and plots it.

    :param input_data_frame: The data frame that you want to plot.
    :type input_data_frame: pd.DataFrame
    :param show: If True, the plot will be shown
    :type bool
    :type input_data_frame: pd.DataFrame

    :return: None
    """
    file_path = "plots/value_over_time/all_data_frames_concatenated.png"

    if show:
        print("Plotting concatenated data... \n")
    else:
        print(f"Plotting concatenated data and saving it to file: \n{file_path}... \n")

    for data in input_data_frame:
        sns.lineplot(
            data=input_data_frame[data],
            x=input_data_frame[data]['Date'],
            y=input_data_frame[data]['Value'],
            dashes=False,
            lw=0.7,
        )

    plt.title('Value over time')
    plt.ylabel('Value in USD ($)')
    plt.legend(labels=input_data_frame.keys(), bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.subplots_adjust(right=0.75)
    plt.savefig(file_path, dpi=300)

    if show:
        plt.show()
    plt.close()


def plot_data_frames_value_over_time(input_data_frames, show=False):
    """
    Plots the data from the Pandas data frames.

    :param input_data_frames: A dictionary of Pandas data frames
    :type input_data_frames: dict
    :param show: If True, the plot will be shown
    :type show: bool

    :return: None
    """
    print('Staring to create line-plots for value over time data... ')

    for data_frame in input_data_frames:
        file_path = f"plots/value_over_time/value_over_time_{data_frame}.png"

        if show:
            print(f'Plotting value over time data for {data_frame}... ')
        else:
            print(f'Plotting value over time data for {data_frame} and saving it to file: \n{file_path}... ')

        sns.lineplot(
            data=input_data_frames[data_frame],
            x='Date',
            y='Value',
            lw=0.5,
            errorbar=None,
            estimator=None,
            color='red',
        )
        plt.title(f'{data_frame}')
        plt.ylabel('Value in USD ($)')

        plt.savefig(file_path, dpi=300)

        if show:
            plt.show()
        plt.close()

    print('Done plotting value over time data. \n')


def calculate_descriptive_statistics(input_data_frames):
    """
    Calculates the descriptive statistics for the Pandas data frames.

    :param input_data_frames: A dictionary of Pandas data frames
    :type input_data_frames: dict

    :return: None
    """
    statistics = {}

    for data_frame in input_data_frames:
        print(f'Calculating descriptive statistics for {data_frame}... ')

        this_df = input_data_frames[data_frame]['Value']

        # Create a Pandas data series with the descriptive statistics
        output_data_frame = pd.Series({'Count': int(this_df.count()),
                                       'Min': this_df.min(),
                                       'Max': this_df.max(),
                                       'Mean': this_df.mean(),
                                       'Median': this_df.median(),
                                       'Std': this_df.std(),
                                       'Var': this_df.var(),
                                       },
                                      index=['Count',
                                             'Min',
                                             'Max',
                                             'Mean',
                                             'Median',
                                             'Std',
                                             'Var']
                                      )

        # Output the descriptive statistics to a csv file
        output_data_frame.to_csv(
            f'./csv/individual_statistics/statistics_{data_frame}.csv',
            header=False
        )

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

    file_path = './tables/summarized_descriptive_statistics.png'
    data = input_data_frame.values
    columns = input_data_frame.columns
    rows = input_data_frame.index

    if show:
        print("Plotting descriptive statistics table... \n")
    else:
        print(f"Plotting descriptive statistics table and saving it to file: \n{file_path}\n")

    table = plt.table(
        cellText=data.round(4),
        cellLoc='left',
        colLabels=columns,
        colLoc='left',
        colColours=['Lightblue'] * len(columns),
        rowLabels=rows,
        rowLoc='left',
        rowColours=['lightyellow'] * len(rows),
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(4)
    table.scale(1.1, 1.2)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(
        file_path,
        facecolor='w',
        edgecolor='w',
        format=None,
        bbox_inches=None,
        orientation='landscape',
        pad_inches=0.05,
        dpi=300
    )

    if show:
        plt.show()
    plt.close()


def plot_correlation_matrix_heatmap(input_correlation_matrix, show=False):
    """
    This function takes in a data frame, creates a correlation matrix,
    and plots a heatmap of the correlation matrix.

    :param input_data_frame: The data frame that contains the data
    that you want to calculate the correlation matrix for
    :type input_data_frame: pd.DataFrame
    :param show: If True, the plot will be shown
    :type show: bool

    :return: Correlation matrix
    :rtype: pd.DataFrame
    """
    file_path = './plots/heatmaps/correlation_heatmap.png'

    if show:
        print("Plotting correlation matrix heatmap... \n")
    else:
        print(f"Plotting correlation matrix heatmap and saving it to file: \n{file_path}\n")

    # Plot heatmap of correlation matrix
    correlation_heatmap = sns.heatmap(
        input_correlation_matrix,
        annot=True,
        annot_kws={'size': 5},
        cmap="flare"
    )
    # Set label sizes
    correlation_heatmap.xaxis.set_tick_params(labelsize=5)
    correlation_heatmap.yaxis.set_tick_params(labelsize=5)

    # Set title, x and y labels
    plt.title('Korrelationsmatris', fontsize=16, pad=20)
    plt.xlabel('Valutor', fontsize=10)
    plt.ylabel('Valutor', fontsize=10)
    plt.text(1, 1, '', ha='center', va='center', fontsize=4)
    plt.xticks(rotation=75, size=6)
    plt.yticks(rotation=15, size=6)
    # plt.subplots_adjust(left=0.25, bottom=0.3)

    # Save figure to file
    correlation_heatmap.figure.savefig(
        file_path,
        bbox_inches='tight',
        dpi=300
    )

    # Show plots
    if show:
        plt.show()
    plt.close()


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
        file_path = f'./plots/normal_distribution/normal_distribution_{data}.png'

        if show:
            print(f"Plotting the normal distribution of {data}...")
        else:
            print(f"Plotting the normal distribution of {data} and saving it to file: "
                  f"\n{file_path}")

        dataset = input_data_frames[data]['Value']

        sns.distplot(
            x=dataset,
            color='purple',
            kde=True,
            kde_kws={'color': 'darkgreen', 'linewidth': 0.6, 'label': 'Kernel Density Estimation', 'shade': True,
                     'alpha': 0.25},
            label=data,
            fit=norm,
            fit_kws={'color': 'red', 'linewidth': 1, 'label': 'Normal Distribution'}
        )
        sns.set_style('whitegrid')
        plt.axvline(dataset.mean(), color='blue', linestyle='--', linewidth=1, label='Mean Value')
        plt.title(f'{data} Normal Distribution', fontsize=20)
        plt.ylabel('Density', fontsize=10)
        plt.xlabel('Value', fontsize=10)
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            file_path,
            bbox_inches='tight',
            dpi=300
        )

        if show:
            plt.show()
        plt.close()

    print("Done plotting normal distributions. \n")


def calculate_linear_regressions(input_data_frames):
    """
    This function takes in a dictionary of data frames,
    and returns a dictionary of linear regression models

    :param input_data_frames: A dictionary of data frames
    :type input_data_frames: dict
    :param show: If True, the plot will be shown. If False, the plot will be
    saved to a file, defaults to False (optional)
    :type show: bool, optional
    """
    print("Starting to calculate simple linear regressions...")

    output_linear_regressions = {}

    for data_frame in input_data_frames:
        print(f"Calculating simple linear regression for {data_frame}...")

        # Fit the linear regression model
        x_data = input_data_frames[data_frame]['Date'].map(datetime.toordinal).values.reshape(-1, 1)
        y_data = input_data_frames[data_frame]['Value'].values.reshape(-1, 1)

        # Create a linear regression model
        model = LinearRegression().fit(x_data, y_data)

        # Get the slope and intercept of the line best fit
        slope = model.coef_[0][0]
        intercept = model.intercept_[0]
        r_squared = model.score(x_data, y_data)
        print(f"R-squared: {r_squared}")

        # Predict the y values
        y_predict = model.predict(x_data)

        # Calculate the linear regression model
        fx = intercept + slope * x_data

        # Calculate the residuals
        residuals = y_data - y_predict

        # Store the linear regression model in a dictionary
        output_linear_regressions[data_frame] = {
            'x': x_data.flatten(),
            'y': y_data.flatten(),
            'prediction': y_predict.flatten(),
            'fx': fx.flatten(),
            'r_squared': r_squared,
            'slope': slope,
            'intercept': intercept,
            'residuals': residuals.flatten(),
            'mse': metrics.mean_squared_error(x_data, y_predict),
            'mae': metrics.mean_absolute_error(x_data, y_predict),
            'rmse': np.sqrt(metrics.mean_squared_error(x_data, y_predict)),
        }

    print("Done calculating linear regressions. \n")

    return output_linear_regressions


def plot_linear_regression_tables(input_data_frames, transformed=False, show=False):
    """
    It takes a dictionary of data frames, and plots a table for each data frame

    :param input_data_frames: A dictionary of data frames
    :type input_data_frames: dict
    :param transformed: If True, the data frames are printed as transformed, defaults to False
    :type transformed: bool, optional
    :param show: If True, the plot will be shown. If False, the plot will be
    saved to a file, defaults to False (optional)
    :type show: bool, optional
    """
    for data_frame in input_data_frames:
        if transformed:
            file_path = f'./tables/linear_regression_tables_transformed/linear_regression_{data_frame}_transformed.png'
            row_color = 'lightblue'
            column_color = 'lightgreen'
        else:
            file_path = f'./tables/linear_regression_tables/linear_regression_data_{data_frame}.png'
            row_color = 'lightblue'
            column_color = 'lightyellow'

        if show:
            print(f"Plotting linear regression table for {data_frame}...")
        else:
            print(f"Plotting linear regression table for {data_frame} and saving it to file: "
                  f"\n{file_path}")

        table_data = pd.Series({
            'slope': input_data_frames[data_frame]['slope'],
            'intercept': input_data_frames[data_frame]['intercept'],
            'r_squared': input_data_frames[data_frame]['r_squared'],
            'mse': input_data_frames[data_frame]['mse'],
            'mae': input_data_frames[data_frame]['mae'],
            'rmse': input_data_frames[data_frame]['rmse'],
        }, name=data_frame)

        sns.set_style('darkgrid')
        plot = plt.table(
            cellText=[table_data.values.round(3)],
            cellLoc='left',
            colLabels=table_data.index,
            colLoc='left',
            colColours=[column_color] * len(table_data),
            rowLabels=[table_data.name],
            rowLoc='left',
            rowColours=[row_color] * len(table_data),
            loc='center',
        )
        plot.auto_set_font_size(False)
        plot.set_fontsize(8)
        plot.scale(1, 1.25)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(file_path, bbox_inches='tight', dpi=300)

        if show:
            plt.show()
        plt.close()

    print("Done plotting linear regression tables. \n")


def plot_linear_regression_graphs(input_data_frames, show=False):
    print("Starting to plot linear regressions... ")

    for data in input_data_frames:
        file_path = f'./plots/linear_regression/linear_regression_{data}.png'

        if show:
            print(f"Plotting linear regression for {data}... ")
        else:
            print(f"Plotting linear regression for {data} and saving it to file... \n{file_path}")

        dataset = pd.DataFrame({
            'Date': input_data_frames[data]['Date'].apply(datetime.toordinal),
            'Value': input_data_frames[data]['Value'].values,
        })

        sns.set_style('whitegrid')
        sns.regplot(
            x='Date',
            y='Value',
            data=dataset,
            ci=95,
            line_kws={'color': 'red', 'lw': 0.75},
            marker='.',
            scatter_kws={'s': 5, 'alpha': 0.75},
        )

        # sns.regplot(
        #     x='Date',
        #     y='Value',
        #     data=dataset,
        #     ci=95,
        #     line_kws={'color': 'green', 'lw': 0.75},
        #     order=2,
        #     scatter=False,
        # )

        plt.title(f'Linear regression for {data}', fontsize=16)
        plt.savefig(file_path, bbox_inches='tight', dpi=300)

        if show:
            plt.show()
        plt.close()

    print('Done calculating simple linear regressions. \n')


def calculate_transformed_regressions(input_data_frames, show=False):
    print("Starting to calculate transformed linear regressions... \n")
    transformed_regression_models = {}

    for data in input_data_frames:
        print(f"Calculating transformed linear regression for {data}... ")

        x = input_data_frames[data]['Date'].map(datetime.toordinal).values.reshape(-1, 1)
        y = input_data_frames[data]['Value'].values.reshape(-1, 1)
        constant = 10
        y_min = np.min(y)

        if y_min < 0:
            y_log = np.log(y - y_min + constant).reshape(-1, 1)
        else:
            y_log = np.log(y).reshape(-1, 1)

        observations_n = len(y_log)
        sigma = 2  # Noise factor
        noise = np.random.normal(1, observations_n)  # Noise vector

        # Linear Model
        linear_model = LinearRegression().fit(x, y)
        r_squared_linear = linear_model.score(x, y)  # R-squared to check for goodness of fit
        intercept = linear_model.intercept_[0]
        slope = linear_model.coef_[0][0]
        y_predict = linear_model.predict(x)
        residuals_linear = y - y_predict

        # Transformed Model
        transformed_model = LinearRegression().fit(x, y_log)
        r_squared_transformed = transformed_model.score(x, y_log)  # R-squared to check for goodness of fit
        intercept_transformed = transformed_model.intercept_[0]
        slope_transformed = transformed_model.coef_[0][0]
        y_predict_log = transformed_model.predict(x)

        # Calculate residuals
        residulas_transformed = y_log - y_predict_log

        # Equation of the transformed regression line
        fx_transformed = intercept_transformed + slope_transformed * x

        transformed_regression_models[data] = {
            'x': x,
            'y': y_log,
            'fx': fx_transformed,
            'prediction': y_predict_log,
            'intercept': intercept_transformed,  # Intercept = a
            'slope': slope_transformed,  # Slope = b
            'r_squared': r_squared_transformed,
            'residuals': residulas_transformed,
            'mse': metrics.mean_squared_error(y_log, y_predict_log),
            'mae': metrics.mean_absolute_error(y_log, y_predict_log),
            'rmse': np.sqrt(metrics.mean_squared_error(y_log, y_predict_log)),
        }

        dataset_linear = pd.DataFrame({
            'Date': x.flatten(),
            'Value': y.flatten(),
        })

        dataset_transformed = pd.DataFrame({
            'Date': x.flatten(),
            'Value': y_log.flatten(),
        })

        sns.regplot(
            x='Date',
            y='Value',
            data=dataset_linear,
            ci=95,
            # order=sigma,
            line_kws={'color': 'red', 'lw': 0.75},
            scatter=True,
            scatter_kws={'s': 10, 'alpha': 0.7},
            marker='.',
        )

        sns.regplot(
            x='Date',
            y='Value',
            data=dataset_transformed,
            ci=95,
            order=sigma,
            scatter=True,
            scatter_kws={'s': 5, 'alpha': 0.7, 'color': 'green'},
            marker='.',
            line_kws={'color': 'orange', 'lw': 0.75},
        )

        plt.title(f'Linear regression for {data}', fontsize=16)
        plt.savefig(f'./plots/linear_regression_transformed/transformed_regression_{data}.png', bbox_inches='tight',
                    dpi=300)

        if show:
            plt.show()
        plt.close()

        print(f"Done calculating transformed linear regression for {data}... ")

    print(f"Done calculating transformed linear regression... \n")

    return transformed_regression_models


def plot_transformed_regression_graphs(input_data_frames, show=False):
    print("Starting to plot transformed linear regressions... ")

    for data in input_data_frames:
        file_path = f'./plots/linear_regression_transformed/linear_regression_transformed_{data}.png'

        if show:
            print(f"Plotting transformed linear regression for {data}... ")
        else:
            print(f"Plotting transformed linear regression for {data} and saving it to file... "
                  f"\n.{file_path}")

        dataset = pd.DataFrame({
            'Date': input_data_frames[data]['y'].flatten(),
            'Value': input_data_frames[data]['x'].flatten(),
        })

        sns.set_style('whitegrid')
        sns.regplot(
            x='Date',
            y='Value',
            data=dataset,
            ci=95,
            line_kws={'color': 'red', 'lw': 0.75},
            marker='.',
            scatter_kws={'s': 5, 'alpha': 0.75},
        )

        plt.title(f'Linear regression for {data}', fontsize=16)
        plt.savefig(file_path, bbox_inches='tight', dpi=300)

        if show:
            plt.show()
        plt.close()


def plot_residuals(input_data_frames, input_transformed_data_frames, show=False):
    print("Starting to plot residuals... ")

    for data in input_data_frames:
        file_path = f'./plots/residuals/residuals_{data}.png'

        if show:
            print(f"Plotting residuals for {data}... ")
        else:
            print(f"Plotting residuals for {data} and saving it to file... "
                  f"\n.{file_path}")

        dataset_linear = pd.DataFrame({
            'Date': input_data_frames[data]['x'].flatten(),
            'Value': input_data_frames[data]['residuals'].flatten(),
        })

        dataset_transformed = pd.DataFrame({
            'Date': input_transformed_data_frames[data]['x'].flatten(),
            'Value': input_transformed_data_frames[data]['residuals'].flatten(),
        })

        sns.set_style('whitegrid')
        sns.residplot(
            data=dataset_linear,
            x='Date',
            y='Value',
            scatter_kws={'s': 5, 'alpha': 0.75, 'color': 'green'},
        )

        sns.residplot(
            data=dataset_transformed,
            x='Date',
            y='Value',
            scatter_kws={'s': 5, 'alpha': 0.75, 'color': 'red'},
            line_kws={'color': 'blue', 'lw': 1, 'alpha': 0.75, 'linestyle': '--'},
        )

        plt.title(f'Residuals for {data}', fontsize=16)
        plt.savefig(file_path, bbox_inches='tight', dpi=300)

        if show:
            plt.show()
        plt.close()


def plot_residual_normal_distributions(input_data_frames, show=False):
    """
    This function plots the normal distribution of each of the input data frames

    :param input_data_frames: A dictionary of data frames
    :type input_data_frames: dict
    :param show: If True, the plot will be shown, defaults to False (optional)
    :type show: bool, optional
    """
    print("Starting to plot normal distributions...")

    for data in input_data_frames:
        file_path = f'./plots/normal_distribution_transformed/transformed_normal_distribution_{data}.png'

        if show:
            print(f"Plotting the normal distribution of transformed {data}...")
        else:
            print(f"Plotting the normal distribution of transformed {data} and saving it to file: "
                  f"\n{file_path}")

        dataset = input_data_frames[data]['residuals']

        sns.distplot(
            x=dataset,
            color='purple',
            kde=True,
            kde_kws={'color': 'darkgreen', 'linewidth': 0.6, 'label': 'Kernel Density Estimation', 'shade': True,
                     'alpha': 0.25},
            label=data,
            fit=norm,
            fit_kws={'color': 'red', 'linewidth': 1, 'label': 'Normal Distribution'}
        )
        sns.set_style('whitegrid')
        plt.axvline(dataset.mean(), color='blue', linestyle='--', linewidth=1, label='Mean Value')
        plt.title(f'{data} Normal Distribution', fontsize=20)
        plt.ylabel('Density', fontsize=10)
        plt.xlabel('Value', fontsize=10)
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            file_path,
            bbox_inches='tight',
            dpi=300
        )

        if show:
            plt.show()
        plt.close()

    print("Done plotting normal distributions. \n")


# Paths for the csv files containing the data
data_paths = {
    'Bitcoin': './csv/coin_Bitcoin.csv',
    'Cardano': './csv/coin_Cardano.csv',
    'ChainLink': './csv/coin_ChainLink.csv',
    'Cosmos': './csv/coin_Cosmos.csv',
    'Dogecoin': './csv/coin_Dogecoin.csv',
    'Ethereum': './csv/coin_Ethereum.csv',
    'Litecoin': './csv/coin_Litecoin.csv',
    'Monero': './csv/coin_Monero.csv',
    'Polkadot': './csv/coin_Polkadot.csv',
    'Solana': './csv/coin_Solana.csv',
    'Stellar': './csv/coin_Stellar.csv',
    'Tether': './csv/coin_Tether.csv',
    'Uniswap': './csv/coin_Uniswap.csv',
    'USDCoin': './csv/coin_USDCoin.csv',
    'XRP': './csv/coin_XRP.csv',
}

# Declare dictionaries to hold data frames
parsed_data_paths = {}  # Dictionary for parsed data paths
data_frames = {}  # Dictionary for Pandas data frames

# Execution of calculations
clean_all_data_frames(data_paths)
concat_data_frame = concatenate_data_frames(data_frames)
correlation_matrix = concat_data_frame.corr()
descriptive_statistics = calculate_descriptive_statistics(data_frames)
descriptive_statistics_table = concatenate_data_frames(descriptive_statistics)
linear_regressions = calculate_linear_regressions(data_frames)
transformed_regressions = calculate_transformed_regressions(data_frames)

# Plot data functions
print("Starting to plot data...")
# plot_concatenated_data_frame(data_frames)
# plot_data_frames_value_over_time(data_frames)
# plot_descriptive_statistics_table(descriptive_statistics_table)
# plot_correlation_matrix_heatmap(correlation_matrix)
# plot_normal_distributions(data_frames)
# plot_linear_regression_tables(linear_regressions)
# plot_linear_regression_graphs(data_frames)
# plot_linear_regression_tables(transformed_regressions, True)
# plot_residuals(linear_regressions, transformed_regressions)
plot_residual_normal_distributions(transformed_regressions, True)
print("Done plotting data... \n")
