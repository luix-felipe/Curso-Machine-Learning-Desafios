from pathlib import Path
import os
import pickle

import cv2
import imutils
import imutils.paths as paths
import numpy as np
import sklearn
import sklearn.preprocessing
import sklearn.svm
from tqdm import tqdm


os.chdir(Path(__file__).resolve().parent)

# Carregar os modelos
protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(
    ["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"]
)
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# Captura os paths do dataset
imagePaths = list(paths.list_images("dataset"))

# Lista de caracteristicas e nomes
knownEmbeddings = []
knownNames = []

# Loop sobre as imagens do dataset
for imagePath in tqdm(imagePaths):
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )
    detector.setInput(imageBlob)
    detections = detector.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(
                face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
            )
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())

# Salva embeddings
data = {"embeddings": knownEmbeddings, "names": knownNames}
os.makedirs("output", exist_ok=True)
f = open("output/embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()

# Treinar classificador
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

le = sklearn.preprocessing.LabelEncoder()
labels = le.fit_transform(data["names"])

recognizer = sklearn.svm.SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open("output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()

# Processar o modelo
image_to_test_path = "test/modelo_2_1.jpeg"

recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

image = cv2.imread(image_to_test_path)
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)),
    1.0,
    (300, 300),
    (104.0, 177.0, 123.0),
    swapRB=False,
    crop=False,
)
detector.setInput(imageBlob)
detections = detector.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        if fW < 20 or fH < 20:
            continue

        faceBlob = cv2.dnn.blobFromImage(
            face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
        )
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(
            image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2
        )
        print(text)

cv2.imwrite("output/desafio_2_resultado.jpeg", image)
print("Imagem salva em: output/desafio_2_resultado.jpeg")
