import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class EconomicForecast:
    def __init__(self, inflation_data, dollar_data):
        self.inflation_data = inflation_data
        self.dollar_data = dollar_data
        self.inflation_model = None
        self.dollar_model = None

    def train_inflation_model(self):
        years = np.arange(len(self.inflation_data)).reshape(-1, 1)
        self.inflation_model = LinearRegression()
        self.inflation_model.fit(years, self.inflation_data)

    def train_dollar_model(self):
        years = np.arange(len(self.dollar_data)).reshape(-1, 1)
        self.dollar_model = LinearRegression()
        self.dollar_model.fit(years, self.dollar_data)

    def predict_inflation(self, years_ahead):
        last_year = len(self.inflation_data)
        future_years = np.arange(last_year, last_year + years_ahead).reshape(-1, 1)
        print(f"Dollar{self.inflation_model.predict(future_years)}")
        return self.inflation_model.predict(future_years)

    def predict_dollar(self, years_ahead):
        last_year = len(self.dollar_data)
        future_years = np.arange(last_year, last_year + years_ahead).reshape(-1, 1)
        return self.dollar_model.predict(future_years)