import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from forecast import EconomicForecast

class RealEstatePricePredictor:
    def __init__(self):
        self.data = []
        self.inflation_values = []
        self.dollar_values = []
        self.df = None
        self.model = None
        self.economic_forecast = None

    def fetch_data(self):
        url = 'https://belrielt.by/Brest/cena/?t=dinamika'
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table')
            rows = table.find_all('tr')
            for row in rows[8:]:
                cols = row.find_all('td')
                if len(cols) > 0:
                    first_column = cols[0].get_text(strip=True)
                    second_column = cols[1].get_text(strip=True).replace(' ', '')
                    self.data.append([first_column, second_column])

    def fetch_inflation_data(self):
        url_ifl = 'https://benefit.by/info/inflyaciya/'
        response_ifl = requests.get(url_ifl)
        if response_ifl.status_code == 200:
            soup_ifl = BeautifulSoup(response_ifl.text, 'html.parser')
            div_table_ifl = soup_ifl.find('div', class_="r-table")
            table_ifl = div_table_ifl.find('table')
            rows_ifl = table_ifl.find_all('tr')
            sums = [0] * 8
            for row in rows_ifl[1:]:
                cols = row.find_all('td')
                if len(cols) >= 8:
                    for i in range(1, 8):
                        value = cols[i].get_text(strip=True).replace(',', '.')
                        try:
                            value = float(value)
                            sums[i - 1] += value
                        except ValueError:
                            pass
            self.inflation_values = sums[:8][::-1]

    def fetch_dollar_data(self):
        self.dollar_values = [0] * 8
        for year in range(2017, 2025):
            url_dol = f'https://etalonline.by/spravochnaya-informatsiya/valuta/arch/{year}/'
            response_dol = requests.get(url_dol)
            if response_dol.status_code == 200:
                soup_dol = BeautifulSoup(response_dol.text, 'html.parser')
                div_table_dol = soup_dol.find('div', class_="left-dop")
                if div_table_dol:
                    tables_dol = div_table_dol.find_all('table')
                    if len(tables_dol) > 1:
                        table_dol = tables_dol[1]
                        rows_dol = table_dol.find_all('tr')
                        if rows_dol:
                            cols_dol = rows_dol[4].find_all('td')
                            if len(cols_dol) > 0:
                                dol = cols_dol[2].get_text(strip=True)
                                try:
                                    dol = float(dol)
                                    self.dollar_values[year - 2017] = dol
                                except ValueError:
                                    pass

    def prepare_data(self):
        self.df = pd.DataFrame(self.data, columns=['Год', 'Цена за м²'])
        self.df['Цена за м²'] = pd.to_numeric(self.df['Цена за м²'], errors='coerce')
        self.df['Год'] = pd.to_numeric(self.df['Год'], errors='coerce')
        self.df.dropna(inplace=True)

        # Создаем экземпляр EconomicForecast и обучаем модели
        self.economic_forecast = EconomicForecast(self.inflation_values, self.dollar_values)
        self.economic_forecast.train_inflation_model()
        self.economic_forecast.train_dollar_model()

        # Добавляем инфляцию и доллар в DataFrame
        self.df['Инфляция'] = self.inflation_values
        self.df['Доллар'] = self.dollar_values

    def train_model(self, model_choice):
        X = self.df[['Год', 'Инфляция', 'Доллар']]
        y = self.df['Цена за м²']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if model_choice == 1:  # Линейная регрессия
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
        elif model_choice == 2:  # Полиномиальная регрессия
            degree = 2  # Степень полинома
            self.model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            self.model.fit(X_train, y_train)
        else:
            print("Неверный выбор модели. Используется линейная регрессия по умолчанию.")
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)

    def predict(self):
        years = np.arange(2017, 2051)
        inflation_predictions = self.economic_forecast.predict_inflation(len(years))
        dollar_predictions = self.economic_forecast.predict_dollar(len(years))

        X_predict = pd.DataFrame({
            'Год': years,
            'Инфляция': inflation_predictions,
            'Доллар': dollar_predictions
        })

        predictions = self.model.predict(X_predict)
        return years, predictions

    def evaluate_model(self):
        X = self.df[['Год', 'Инфляция', 'Доллар']]
        y = self.df['Цена за м²']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mse, r2

    def calculation_house(self, years, predictions, zp, ch, gp, km, ip, proc):
        start_year = gp
        index = gp - years[0]
        if index < 0 or index >= len(predictions):
            raise ValueError("Индекс выходит за пределы предсказаний.")
        price = predictions[index] * km


        ip_sum = 0
        days_for_ip = ip * 365
        day_pl = price / days_for_ip
        for day in range(0, days_for_ip):
            ip_sum += day_pl + day_pl * proc
            price += day_pl * proc
            day_pl = price / days_for_ip
        m_pl = price / (ip*12)
        if (zp*ch) < m_pl:
            print(f"остаток после платежа:{zp*ch-m_pl}")
        else:
            print(f"Недостаточно {m_pl-zp*ch} BYN для платежа")
        return price, ip_sum, m_pl

    def plot_results(self, years, predictions):
        plt.figure(figsize=(14, 7))
        plt.plot(years, predictions, label='Предсказанная цена', color='blue')
        plt.scatter(self.df['Год'], self.df['Цена за м²'], color='red', label='Фактические данные')
        plt.xlabel('Год')
        plt.ylabel('Цена за м²')
        plt.title('Предсказанная цена по годам')
        plt.legend()
        plt.grid()
        plt.show()