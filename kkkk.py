import cv2
from matplotlib import pyplot as plt

# Открываем изображение
img = cv2.imread("people.png")

# OpenCV открывает изображения в цветовой схеме BGR,
# но нам нужна цветная (на будущее) и черно-белая версии
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Загружаем в программу данные каскада Хаара
face_data = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

faces = face_data.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
print(faces)

for (x, y, width, height) in faces:
    cv2.circle(img_rgb, (x + (width // 2), y + (height // 2)), width // 2, (0, 255, 0), 5)

plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()