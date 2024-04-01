from virtual_background import process_image
import cv2
import numpy as np

# тестовое изображение и фон
IMAGE = 'datasets/obj2.jpg'
BACKGROUND = 'background/back4.jpg'

# преобразуем данные в байтовый формат
# в таком виде получаем данные с вебки
data_img = cv2.imread(IMAGE)
data_img_bytes = cv2.imencode('.jpg', data_img)[1].tobytes()

back_img = cv2.imread(BACKGROUND)
back_img_bytes = cv2.imencode('.jpg', back_img)[1].tobytes()

# преобразуем данные из байтов в обычные
processed_img_bytes = process_image(data_img_bytes, back_img_bytes)
nparr = np.frombuffer(processed_img_bytes, np.byte)
processed_img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

# локальное сохранение
cv2.imwrite('processed_images/image.jpg', processed_img)