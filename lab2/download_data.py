import os
import requests



# Ссылка на файл .csv в репозитории Git
url = 'https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/video_games_sales.csv'

# Загрузка файла .csv
response = requests.get(url)

# Проверка успешности загрузки
if response.status_code == 200:
    # Создание папки для сохранения файла, если её ещё нет
    if not os.path.exists('data'):
        os.makedirs('data')

    # Сохранение файла .csv
    with open('data/data.csv', 'wb') as f:
        f.write(response.content)

    print("Данные успешно сохранены.")
else:
    print("Ошибка загрузки файла.")

# pip freeze > requirements.txt