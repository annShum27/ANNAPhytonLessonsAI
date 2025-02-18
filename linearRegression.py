import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Данные: площадь квартиры и её цена
X = np.array([[30], [50], [70], [90], [120]])  # Площадь
y = np.array([2.5, 4.0, 5.5, 7.0, 9.0])  # Цена

# Создание модели линейной регрессии
model = LinearRegression()

# Обучение модели
model.fit(X, y)

# Предсказания: делаем предсказания для тех же данных
y_pred = model.predict(X)

# Визуализация: график площади vs. цены
plt.scatter(X, y, color='blue')  # Точки, где реальные данные
plt.plot(X, y_pred, color='red')  # Линия регрессии (предсказания)
plt.xlabel('Площадь (м²)')
plt.ylabel('Цена (тыс. рублей)')
plt.title('Предсказание цен на квартиры')
plt.show()

# Выводим предсказания
print("Предсказанные цены:", y_pred)