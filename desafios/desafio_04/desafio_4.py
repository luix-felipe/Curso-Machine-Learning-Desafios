from pathlib import Path
import os

import cv2
import torch
import ultralytics


os.chdir(Path(__file__).resolve().parent)

# Baixa o modelo de pose da YOLO
model = ultralytics.YOLO("yolo11n-pose.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Processa a imagem
results = model("desafio_4.jpeg")

print(results[0])

# Gera visualizacao do resultado
im_array = results[0].plot()
rgb_img = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

output_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("desafio_4_resultado.jpeg", output_img)

print("Imagem salva em: desafio_4_resultado.jpeg")
