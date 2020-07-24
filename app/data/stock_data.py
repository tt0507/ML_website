import os
import psycopg2
from dotenv import load_dotenv
from app.web_scraping.stock_ws import get_stock_price

# load environment variables
load_dotenv()

url = os.environ["DATABASE_URL"]


def process():
    """
    insert stock price data into postgres server
    :return:
    """
    with psycopg2.connect(url, sslmode='require') as connect:
        with connect.cursor() as postgres:
            postgres.execute("DELETE FROM ml_website.dow_jones_industrial")
            postgres.execute("DELETE FROM ml_website.sp500")
            postgres.execute("DELETE FROM ml_website.nasdaq")

            # get data
            data = get_stock_price()
            dji_df = data[0]
            sp500_df = data[1]
            nasdaq_df = data[2]

            # sort data by descending
            dji_df = dji_df.sort_index(ascending=False)
            sp500_df = sp500_df.sort_index(ascending=False)
            nasdaq_df = nasdaq_df.sort_index(ascending=False)

            # change date format into string
            dji_df['Date'] = change_date_format(dji_df)
            sp500_df['Date'] = change_date_format(sp500_df)
            nasdaq_df['Date'] = change_date_format(nasdaq_df)

            # command to insert data into postgres server
            dji_command = "INSERT INTO ml_website.dow_jones_industrial VALUES (%s, %s, %s, %s, %s, %s, %s)"
            sp500_command = "INSERT INTO ml_website.sp500 VALUES (%s, %s, %s, %s, %s, %s, %s)"
            nasdaq_command = "INSERT INTO ml_website.nasdaq VALUES (%s, %s, %s, %s, %s, %s, %s)"

            # insert data into postgres server
            for num in range(len(dji_df)):
                # dji
                dji = dji_df.iloc[num].values
                # date, open_price, high_price, low_price, close_price, adj_close_price, volume
                price_data = (dji[0], dji[1].item(), dji[2].item(), dji[3].item(), dji[4].item(),
                              dji[5].item(), int(dji[6].item()))
                postgres.execute(dji_command, price_data)

                # sp500
                sp500 = sp500_df.iloc[num].values
                # date, open_price, high_price, low_price, close_price, adj_close_price, volume
                price_data = (sp500[0], sp500[1].item(), sp500[2].item(), sp500[3].item(), sp500[4].item(),
                              sp500[5].item(), int(sp500[6].item()))
                postgres.execute(sp500_command, price_data)

                nasdaq = nasdaq_df.iloc[num].values
                # date, open_price, high_price, low_price, close_price, adj_close_price, volume
                price_data = (nasdaq[0], sp500[1].item(), nasdaq[2].item(), nasdaq[3].item(), nasdaq[4].item(),
                              nasdaq[5].item(), int(nasdaq[6].item()))
                postgres.execute(nasdaq_command, price_data)


def change_date_format(df):
    """
    change format of date
    :param df: dataframe in which date column wants to be modified
    :return: array of date in string format
    """
    dates = []
    for date in df['Date'].values:
        formatted_date = date.strftime('%Y-%m-%d')
        dates.append(formatted_date)

    return dates


if __name__ == "__main__":
    process()
