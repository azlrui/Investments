from datetime import datetime, date

import request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Stock:
    def __init__(self, name=None, permco=None):
        self.name = name
        self.permco = permco

    def download_data(self, initial_date, final_date):

        query = f"select permco, date, ret " + "from crsp.dsf " + f"where permco in ({self.permco}) " + f"and date>='{initial_date}' and date<='{final_date}'"
        query = db.raw_sql(query)

        table = self.__create_df(query)

        return table

    def __create_df(self, df):

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        df['ret'] = (df['ret']+1).cumprod()
        df = df.rename(columns = {'ret':f"{self.name}"})
        df = df.drop(columns=['permco'])

        return df

def download_bundle_stock(stocks, initial_date, final_date):
    liste = []

    for stock in stocks:
        liste.append(stock.download_data(initial_date, final_date))

    final = pd.concat(liste, axis=1)

    return final

def modify_time_series(df, timings):
    final = {}
    for key, value in timings.items():
        #take the final value of each period
        new = df.resample(value).last()
        #recompute the returns according to the new sampling
        final[key] = (new/new.shift(1) - 1).dropna()
    return final

def summary_statistics(timeseries):
    for key, value in timeseries.items():
        desc = value.describe()
        desc = desc.apply(format_rows, axis=1)
        print(f"Summary statistics for {key} frequency:\n")
        print(desc)

def format_rows(row):
    if row.name in ['mean', 'std', 'min', 'max', '25%', '50%', '75%']:  # More precision for mean & std
        return row.apply(lambda x: f"{x*100:.4f}%")
    else:  # Default 2 decimals for other rows
        return row.apply(lambda x: f"{x:.2f}")

def annualized_statistics(time_series, N_periods):
    for key, value in time_series.items():
        desc = value.apply(lambda col: custom_metrics(col, N_periods[key])).T
        print(f"Annualized statistics for {key} frequency:")
        print(desc)


def custom_metrics(x, per):
    return pd.Series({
        'Ann. mean' : f"{x.mean() * per * 100:.2f}%",
        'Ann. std' : f"{x.std() * np.sqrt(per) * 100:.2f}%",
    })

def time_series_plot(time_series):
    for key, value in time_series.items():
        value.plot(label=key,title='Stock returns at '+key+' frequency')
    plt.show()

def rolled_timeseries_mean(timeseries, n_periods, n_min_periods):
    for key, df in timeseries.items():
        timeseries[key] = df.rolling(window = n_periods[key], min_periods= n_min_periods[key]).mean()
        timeseries[key] = timeseries[key] * n_periods[key]

    stocks = list(timeseries.values())[0].columns.tolist()

    for stock in stocks:
        for key, df in timeseries.items():
            plt.plot(df.index, df[stock], label = key)

        plt.title(f"{stock}: Rolling Mean Across Frequencies")

        #Set up y-axis
        y_values = np.arange(-2, 3.1, 0.5)
        plt.ylim(-2, 3)
        plt.yticks(y_values)

        plt.xlabel("Date")
        plt.ylabel("Rolling Mean")
        plt.grid(True)
        plt.legend(title="Frequency")

        plt.show()

def rolled_timeseries_std(timeseries, n_periods, n_min_periods):
    for key, df in timeseries.items():
        timeseries[key] = df.rolling(window = n_periods[key], min_periods= n_min_periods[key]).std()
        timeseries[key] = timeseries[key] * np.sqrt(n_periods[key])

    stocks = list(timeseries.values())[0].columns.tolist()


    for stock in stocks:
            for key in timeseries.keys():
                for column in timeseries[key]:
                    if column == stock:
                        plt.plot(timeseries[key].index, timeseries[key][stock], label = key)

            plt.title(f"{stock}: Rolling Std.Dev Across Frequencies")

            # Set up y-axis
            y_values = np.arange(0, 1.7, 0.2)
            plt.ylim(0, 1.75)
            plt.yticks(y_values)

            plt.xlabel("Date")
            plt.ylabel("Rolling Mean")
            plt.grid(True)
            plt.legend(title="Frequency")

            plt.show()
#request()

#Open it
data = pd.read_csv('data.csv', sep = ';', index_col = ['date'])
data.index = pd.to_datetime(data.index)

#Generate time-series
timings = {'Daily' : 'D', 'Weekly' : 'W', 'Monthly' : 'M'}
time_series = modify_time_series(data, timings)
#Generate time series plot
time_series_plot(time_series)


#Generate summary_statistics
summary_statistics(time_series)

###Generate annualized_summary_statistics
#Define the periods
n_periods = {'Daily':252, 'Weekly':52, 'Monthly':12}
n_min_periods = {'Daily':100, 'Weekly':50, 'Monthly':10}

#Summary of annualized statistics
annualized_statistics(time_series, n_periods)

#Produce figures for mean and std.dev
rolled_timeseries_mean(time_series.copy(), n_periods, n_min_periods)
rolled_timeseries_std(time_series.copy(), n_periods, n_min_periods)