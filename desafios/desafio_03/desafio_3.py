from pathlib import Path
import os

import cv2
import torch
import ultralytics


os.chdir(Path(__file__).resolve().parent)

# Baixa o modelo
model = ultralytics.YOLO("yolo11n.pt")

# Joga o modelo para o dispositivo de computacao
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Processa o video
results = model.track(source="video.mov", conf=0.1, iou=0.7)

# Validar resultados
for r in results:
    print(r)

# Cria o escritor do video de output
writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 20, (2048, 1080))

# Escreve o video de output
for r in results:
    writer.write(r.plot())

writer.release()
print("Video salvo em: output.avi")
