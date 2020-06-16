import os
import sys
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from datetime import datetime
from app.web_scraping.stock_ws import get_stock_price

# load environment variables
load_dotenv()

url = os.environ.get("MYSQL_URL")
database = create_engine(url)
mysql = database.connect()
mysql.execute("use ml_website_database;")

dji = mysql.execute("SELECT * FROM ml_website_database.dow_jones_industrial").fetchall()
sp500 = mysql.execute("SELECT * FROM ml_website_database.sp500").fetchall()
nasdaq = mysql.execute("SELECT * FROM ml_website_database.nasdaq").fetchall()


def process():
    """
    insert stock price into mysql server
    :return:
    """
    if len(dji) <= 1200 and len(sp500) <= 1200 and len(nasdaq) <= 1200:
        mysql.execute("DELETE FROM ml_website_database.dow_jones_industrial")
        mysql.execute("DELETE FROM ml_website_database.sp500")
        mysql.execute("DELETE FROM ml_website_database.nasdaq")

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

        # command to insert data into mysql server
        dji_command = "INSERT INTO ml_website_database.dow_jones_industrial VALUES (%s, %s, %s, %s, %s, %s, %s)"
        sp500_command = "INSERT INTO ml_website_database.sp500 VALUES (%s, %s, %s, %s, %s, %s, %s)"
        nasdaq_command = "INSERT INTO ml_website_database.nasdaq VALUES (%s, %s, %s, %s, %s, %s, %s)"

        # insert data into mysql server
        insert_data(dji_df, dji_command)
        insert_data(sp500_df, sp500_command)
        insert_data(nasdaq_df, nasdaq_command)
    else:
        pass


def change_date_format(df):
    """

    :param df: dataframe in which date column wants to be modified
    :return: array of date in string format
    """
    dates = []
    for date in df['Date'].values:
        formatted_date = date.strftime('%Y-%m-%d')
        dates.append(formatted_date)

    return dates


def insert_data(df, command):
    """

    :param df:
    :param command:
    :return:
    """
    for num in range(len(df)):
        arr = df.iloc[num].values
        date, open_price, high_price, low_price, close_price, adj_close_price, volume = arr[0], arr[1].item(), arr[
            2].item(), arr[3].item(), arr[4].item(), arr[5].item(), int(arr[6].item())
        price_data = (date, open_price, high_price, low_price, close_price, adj_close_price, volume)
        mysql.execute(command, price_data)


if __name__ == "__main__":
    process()
