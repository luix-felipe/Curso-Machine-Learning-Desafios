from pathlib import Path
import os

import cv2


os.chdir(Path(__file__).resolve().parent)

# Ler imagem
image_bgr = cv2.imread("desafio_1.jpeg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Transformar em escala de cinza
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Detectar as faces
detec = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face = detec.detectMultiScale(image_gray, 1.3, 3)
copy_image_rgb = image_rgb.copy()

padding = 100

for (x, y, larg, alt) in face:
    ret = cv2.rectangle(copy_image_rgb, (x, y), (x + larg, y + alt), (0, 255, 0), 3)
    y, height = max(0, y - padding), min(y + alt + padding, image_rgb.shape[0])
    x, width = max(0, x - padding), min(x + larg + padding, image_rgb.shape[1])
    face_img = image_rgb[y:height, x:width, :]

# Borrar a imagem inteira
blured_image = cv2.GaussianBlur(image_rgb, (15, 15), 20)

# Substituir imagem original da face na imagem borrada
final_image = blured_image.copy()
final_image[y:height, x:width, :] = face_img

final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("desafio_1_resultado.jpeg", final_image_bgr)

print("Imagem salva em: desafio_1_resultado.jpeg")
