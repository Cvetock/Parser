import sys
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt6.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from real_estate_predictor import RealEstatePricePredictor

# Класс Worker для выполнения длительных операций в отдельном потоке
class Worker(QtCore.QThread):
    # Сигналы для передачи результатов и ошибок обратно в основной поток
    result_ready = QtCore.pyqtSignal(float, float, float, float, float, list, list)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, predictor, zp, ch, gp, km, ip, proc, model_choice):
        super().__init__()
        self.predictor = predictor  # Экземпляр класса предсказателя
        self.zp = zp  # Зарплата
        self.ch = ch  # Часть зарплаты для ипотеки
        self.gp = gp  # Год покупки квартиры
        self.km = km  # Квадратные метры квартиры
        self.ip = ip  # Срок ипотеки
        self.proc = proc  # Процент по ипотеке
        self.model_choice = model_choice  # Выбор модели

    def run(self):
        try:
            # Получение данных
            self.predictor.fetch_data()
            self.predictor.fetch_inflation_data()
            self.predictor.fetch_dollar_data()
            self.predictor.prepare_data()
            # Обучение модели
            self.predictor.train_model(self.model_choice)
            # Прогнозирование
            years, predictions = self.predictor.predict()

            # # Отладочный вывод
            # print(f"Годы: {years}, Длина: {len(years)}")
            # print(f"Предсказания: {predictions}, Длина: {len(predictions)}")

            mse, r2 = self.predictor.evaluate_model()
            # Вызов метода расчета дома
            price, total_ip_sum, m_pl = self.predictor.calculation_house(years, predictions, self.zp, self.ch, self.gp,
                                                                         self.km, self.ip, self.proc)

            # Отправка результатов обратно в основной поток
            self.result_ready.emit(price, total_ip_sum, m_pl, mse, r2, years, predictions)
        except Exception as e:
            self.error_occurred.emit(str(e))

# Основной класс приложения
class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.predictor = RealEstatePricePredictor()  # Создание экземпляра предсказателя
        self.model_choice = None  # Инициализация переменной выбора модели
        self.initUI()  # Инициализация пользовательского интерфейса

    def initUI(self):
        self.setWindowTitle('Бездомный')  # Заголовок окна
        self.setGeometry(100, 100, 800, 600)  # Размеры окна

        # Создание полей ввода
        self.zp_input = QLineEdit(self)  # Поле для ввода зарплаты
        self.ch_input = QLineEdit(self)  # Поле для ввода части зарплаты для ипотеки
        self.gp_input = QLineEdit(self)  # Поле для ввода года покупки квартиры
        self.km_input = QLineEdit(self)  # Поле для ввода квадратных метров квартиры
        self.ip_input = QLineEdit(self)  # Поле для ввода срока ипотеки
        self.proc_input = QLineEdit(self)  # Поле для ввода процента по ипотеке

        # Установка валидаторов для ввода
        self.zp_input.setValidator(QDoubleValidator())  # Валидатор для зарплаты
        self.ch_input.setValidator(QDoubleValidator(0.0, 100.0, 2))  # Валидатор для части зарплаты
        self.gp_input.setValidator(QDoubleValidator())  # Валидатор для года
        self.km_input.setValidator(QDoubleValidator())  # Валидатор для квадратных метров
        self.ip_input.setValidator(QDoubleValidator())  # Валидатор для срока ипотеки
        self.proc_input.setValidator(QDoubleValidator(0.0, 100.0, 2))  # Валидатор для процента

        # Создание кнопок
        self.predict_button = QPushButton('Прогнозировать', self)  # Кнопка для прогноза
        self.regenerate_button = QPushButton('Перегенерировать прогноз', self)  # Кнопка для перегенерации прогноза
        self.model_button = QPushButton('Выбрать модель', self)  # Кнопка для выбора модели
        self.predict_button.clicked.connect(self.on_predict)  # Подключение обработчика нажатия
        self.regenerate_button.clicked.connect(self.on_regenerate)  # Подключение обработчика перегенерации
        self.model_button.clicked.connect(self.on_model_select)  # Подключение обработчика выбора модели

        # Создание макета для ввода данных
        input_layout = QVBoxLayout()
        input_layout.addWidget(QLabel('Введите вашу з/п (в BYN):'))  # Подпись для зарплаты
        input_layout.addWidget(self.zp_input)  # Поле для зарплаты
        input_layout.addWidget(QLabel('Введите часть зарплаты из которой вы можете оплачивать ипотеку (в %):'))  # Подпись
        input_layout.addWidget(self.ch_input)  # Поле для части зарплаты
        input_layout.addWidget(QLabel('Введите год покупки квартиры:'))  # Подпись
        input_layout.addWidget(self.gp_input)  # Поле для года
        input_layout.addWidget(QLabel('Введите квадратные метры квартиры (в метрах кв):'))  # Подпись
        input_layout.addWidget(self.km_input)  # Поле для квадратных метров
        input_layout.addWidget(QLabel('Введите на сколько лет ипотека, если ипотеки нету введите 0:'))  # Подпись
        input_layout.addWidget(self.ip_input)  # Поле для срока ипотеки
        input_layout.addWidget(QLabel('Введите процент по ипотеке (в %):'))  # Подпись
        input_layout.addWidget(self.proc_input)  # Поле для процента
        input_layout.addWidget(self.model_button)  # Кнопка выбора модели
        input_layout.addWidget(self.predict_button)  # Кнопка прогноза
        input_layout.addWidget(self.regenerate_button)  # Кнопка перегенерации прогноза

        # Создание области для графика
        self.figure = Figure()  # Создание фигуры для графика
        self.canvas = FigureCanvas(self.figure)  # Создание канваса для отображения графика

        # Создание макета для вывода данных
        self.output_label = QLabel(self)  # Метка для вывода результатов
        self.output_label.setWordWrap(True)  # Автоматический перенос текста

        # Основной макет
        main_layout = QHBoxLayout()
        main_layout.addLayout(input_layout)  # Добавление макета ввода
        main_layout.addWidget(self.canvas)  # Добавление области для графика
        main_layout.addWidget(self.output_label)  # Добавление области для вывода результатов

        self.setLayout(main_layout)  # Установка основного макета

    def on_model_select(self):
        # Открытие диалогового окна для выбора модели
        model_choice, ok = QtWidgets.QInputDialog.getItem(self, "Выбор модели", "Выберите модель:", ["Линейная регрессия", "Полиномиальная регрессия"], 0, False)
        if ok and model_choice:
            self.model_choice = 1 if model_choice == "Линейная регрессия" else 2  # Сохранение выбора модели

    def on_predict(self):
        # Проверка, выбрана ли модель
        if self.model_choice is None:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, выберите модель перед прогнозированием.")
            return

        try:
            # Получение данных из полей ввода
            zp = float(self.zp_input.text())
            ch = float(self.ch_input.text())
            gp = int(self.gp_input.text())
            km = float(self.km_input.text())
            ip = int(self.ip_input.text())
            proc = float(self.proc_input.text()) / 100

            # Создание и запуск рабочего потока
            self.worker = Worker(self.predictor, zp, ch, gp, km, ip, proc, self.model_choice)
            self.worker.result_ready.connect(self.on_result_ready)  # Подключение сигнала для получения результатов
            self.worker.error_occurred.connect(self.on_error_occurred)  # Подключение сигнала для обработки ошибок
            self.worker.start()  # Запуск рабочего потока

        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, введите корректные данные.")  # Обработка ошибок ввода

    def on_regenerate(self):
        # Очистка данных в предсказателе
        self.predictor.data = []
        self.predictor.inflation_values = []
        self.predictor.dollar_values = []
        self.predictor.df = None
        self.predictor.model = None

        # Перегенерация прогноза
        self.on_predict()  # Просто повторяем вызов on_predict для перегенерации

    def on_result_ready(self, price, total_ip_sum, m_pl, mse, r2, years, predictions):
        # Проверка на совпадение длин
        if len(years) != len(predictions):
            QMessageBox.critical(self, "Ошибка", "Длина годов и предсказаний не совпадает.")
            return

        # Логика для вывода результатов
        available_funds = float(self.zp_input.text()) * float(self.ch_input.text()) / 100
        if available_funds < m_pl:
            remaining = m_pl - available_funds
            self.output_label.setText(
                f"Полная цена квартиры: {price:.2f} BYN\n"
                f"Общая сумма по ипотеке: {total_ip_sum:.2f} BYN\n"
                f"Ежемесячный платеж: {m_pl:.2f} BYN\n"
                f"Недостаточно {remaining:.2f} BYN для платежа\n"
                f"MSE: {mse:.2f}\nR²: {r2:.2f}"
            )
        else:
            remaining = available_funds - m_pl
            self.output_label.setText(
                f"Полная цена квартиры: {price:.2f} BYN\n"
                f"Общая сумма по ипотеке: {total_ip_sum:.2f} BYN\n"
                f"Ежемесячный платеж: {m_pl:.2f} BYN\n"
                f"Остаток после платежа: {remaining:.2f} BYN\n"
                f"MSE: {mse:.2f}\nR²: {r2:.2f}"
            )

        self.plot_results(years, predictions)  # Построение графика

    def on_error_occurred(self, error_message):
        # Обработка ошибок
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {error_message}")

    def plot_results(self, years, predictions):
        # Построение графика
        self.figure.clear()  # Очистка фигуры
        ax = self.figure.add_subplot(111)  # Добавление подграфика
        ax.plot(years, predictions, label='Предсказанная цена', color='blue')  # Построение линии предсказаний
        ax.set_xlabel('Год')  # Подпись оси X
        ax.set_ylabel('Цена за м²')  # Подпись оси Y
        ax.set_title('Предсказанная цена по годам')  # Заголовок графика
        ax.legend()  # Отображение легенды
        ax.grid()  # Включение сетки
        self.canvas.draw()  # Обновление канваса

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)  # Создание экземпляра приложения
    ex = App()  # Создание экземпляра основного класса
    ex.show()  # Отображение окна
    sys.exit(app.exec())  # Запуск основного цикла приложения