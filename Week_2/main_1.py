import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wrds
from scipy.stats import norm

db = wrds.Connection(wrds_username='razevedo')


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
        df = df.rename(columns={'ret': f"{self.name}"})
        df = df.drop(columns=['permco'])

        return df


initial_date = "2000-01-01"
final_date = "2024-12-31"

if __name__ == '__main__':
    AAPL = Stock("aapl", 7)
    GAMESTOP = Stock("gamestop", 42775)
    TESLA = Stock("tesla", 53453)
    GE = Stock("ge", 20792)
    PG = Stock("pg", 21446)

    aapl = AAPL.download_data(initial_date, final_date)
    gamestop = GAMESTOP.download_data(initial_date, final_date)
    tesla = TESLA.download_data(initial_date, final_date)
    ge = GE.download_data(initial_date, final_date)
    pg = PG.download_data(initial_date, final_date)

    ###Generate a table with all the series
    # Generate the index
    date_index = pd.date_range(start=initial_date, end=final_date, freq="D")
    # Create an empty dataset
    df = pd.DataFrame(index=date_index)
    # Create a unique table
    df = df.join(aapl)
    df = df.join(gamestop)
    df = df.join(tesla)
    df = df.join(ge)
    df = df.join(pg)

    stocks = ["aapl", "gamestop", "tesla", "ge", "pg"]

    # Get rid of extreme values
    df_wind = df.copy()
    for stock in stocks:
        df_wind.loc[np.abs(df_wind[stock]) >= 0.04, stock] = np.nan

    summary_table = pd.DataFrame({
        "mean": df[stocks].mean(),
        "variance": df[stocks].var(),
        "mean_wind": df_wind[stocks].mean(),
        "var_wind": df_wind[stocks].var()
    })

    print(summary_table)
    charac = {"aapl": {}, "gamestop": {}, "tesla": {}, "ge": {}, "pg": {}}
    ###Plot a normal distribution for each stock + add VaR
    for S in stocks:
        # Plot the empirical density function of stock returns
        df[S].plot(kind="hist", bins=500, density=True, label=f"empirical density")

        # Plot the normal distribution with the empirical mean and variance
        mu = df[S].mean()
        sigma = df[S].std()

        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
        y = norm.pdf(x, loc=mu, scale=sigma)
        plt.plot(x, y, label=f"normal pdf calibrated to empirical density", linewidth=2)

        ##Add the VaR
        # VaR if it was normal
        for epsilon in [0.95, 0.99]:
            VaR = mu + sigma * norm.ppf(1 - epsilon)
            charac[S][f'Var_{epsilon}_Normal'] = VaR

        # ES if it was normal
        for epsilon in [0.95, 0.99]:
            charac[S][f'ES_{epsilon}_Normal'] = norm.ppf(1 - epsilon, df[S].mean(), df[S].std())

        # VaR if it empirical
        for epsilon in [0.95, 0.99]:
            q = df[S].dropna().quantile(1 - epsilon)
            charac[S][f'Var_{epsilon}'] = q

        # ES if it was normal
        for epsilon in [0.95, 0.99]:
            charac[S][f'ES_{epsilon}'] = df[S].mean() - df[S].std() / (1 - epsilon) * norm.pdf(
                norm.ppf(1 - epsilon, 0, 1))

        # Plot the normal distribution with the empirical mean_wind and variance_wind
        mu = df_wind[S].mean()
        sigma = df_wind[S].std()

        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
        y = norm.pdf(x, loc=mu, scale=sigma)
        plt.plot(x, y, label=f"normal pdf calibrated to 'winsorized' empirical density", linewidth=2)

        plt.legend(fontsize=5)
        plt.title(S)

        plt.show()

    print(pd.DataFrame(charac))
