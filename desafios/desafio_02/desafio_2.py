from argparse import ArgumentParser
from pathlib import Path
import pickle

import cv2
import numpy as np

try:
    from sklearn import preprocessing, svm
except ImportError as exc:
    preprocessing = None
    svm = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent


def ensure_sklearn() -> None:
    if SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "scikit-learn nao esta instalado. Rode: pip install scikit-learn"
        ) from SKLEARN_IMPORT_ERROR


def resize_width(image, width: int):
    h, w = image.shape[:2]
    scale = width / float(w)
    return cv2.resize(image, (width, int(h * scale)))


def list_images(dataset_dir: Path):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        path for path in dataset_dir.rglob("*") if path.suffix.lower() in extensions
    )


def load_models(detector_dir: Path, embedder_path: Path):
    proto_path = detector_dir / "deploy.prototxt"
    model_path = detector_dir / "res10_300x300_ssd_iter_140000.caffemodel"

    for path in (proto_path, model_path, embedder_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo necessario nao encontrado: {path}\n"
                "Para este desafio, adicione o modelo OpenFace "
                "openface_nn4.small2.v1.t7 na pasta do desafio ou informe "
                "o caminho com --embedder."
            )

    detector = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
    embedder = cv2.dnn.readNetFromTorch(str(embedder_path))
    return detector, embedder


def detect_faces(image, detector, min_confidence: float):
    h, w = image.shape[:2]
    image_blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )
    detector.setInput(image_blob)
    detections = detector.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < min_confidence:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        start_x, start_y, end_x, end_y = box.astype("int")
        start_x, start_y = max(0, start_x), max(0, start_y)
        end_x, end_y = min(w, end_x), min(h, end_y)
        face = image[start_y:end_y, start_x:end_x]
        face_h, face_w = face.shape[:2]
        if face_w < 20 or face_h < 20:
            continue

        faces.append((face, (start_x, start_y, end_x, end_y), float(confidence)))

    return faces


def extract_embedding(face, embedder):
    face_blob = cv2.dnn.blobFromImage(
        face,
        1.0 / 255,
        (96, 96),
        (0, 0, 0),
        swapRB=True,
        crop=False,
    )
    embedder.setInput(face_blob)
    return embedder.forward()


def train_recognizer(
    dataset_dir: Path,
    output_dir: Path,
    detector,
    embedder,
    min_confidence: float,
    resize_to: int,
) -> None:
    ensure_sklearn()

    known_embeddings = []
    known_names = []

    for image_path in list_images(dataset_dir):
        name = image_path.parent.name
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        image = resize_width(image, resize_to)
        faces = detect_faces(image, detector, min_confidence)
        if not faces:
            print(f"Nenhuma face detectada em: {image_path}")
            continue

        face, _, _ = max(faces, key=lambda item: item[0].shape[0] * item[0].shape[1])
        vec = extract_embedding(face, embedder)
        known_names.append(name)
        known_embeddings.append(vec.flatten())

    if not known_embeddings:
        raise RuntimeError("Nenhum embedding foi gerado a partir do dataset.")

    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(known_names)

    recognizer = svm.SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(known_embeddings, labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "embeddings.pickle").open("wb") as file:
        pickle.dump({"embeddings": known_embeddings, "names": known_names}, file)
    with (output_dir / "recognizer.pickle").open("wb") as file:
        pickle.dump(recognizer, file)
    with (output_dir / "le.pickle").open("wb") as file:
        pickle.dump(label_encoder, file)

    print(f"Treino concluido com {len(known_embeddings)} imagens.")


def recognize_image(
    image_path: Path,
    output_path: Path,
    output_dir: Path,
    detector,
    embedder,
    min_confidence: float,
    resize_to: int,
) -> None:
    recognizer_path = output_dir / "recognizer.pickle"
    label_path = output_dir / "le.pickle"
    for path in (recognizer_path, label_path):
        if not path.exists():
            raise FileNotFoundError(f"Modelo treinado nao encontrado: {path}")

    with recognizer_path.open("rb") as file:
        recognizer = pickle.load(file)
    with label_path.open("rb") as file:
        label_encoder = pickle.load(file)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel carregar a imagem: {image_path}")

    image = resize_width(image, resize_to)
    faces = detect_faces(image, detector, min_confidence)

    for face, (start_x, start_y, end_x, end_y), _ in faces:
        vec = extract_embedding(face, embedder)
        preds = recognizer.predict_proba(vec)[0]
        label_index = int(np.argmax(preds))
        probability = preds[label_index]
        name = label_encoder.classes_[label_index]

        text = f"{name}: {probability * 100:.2f}%"
        text_y = start_y - 10 if start_y - 10 > 10 else start_y + 10
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        cv2.putText(
            image,
            text,
            (start_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2,
        )
        print(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Nao foi possivel salvar a imagem em: {output_path}")

    print(f"Imagem salva em: {output_path}")
    print(f"Faces detectadas: {len(faces)}")


def parse_args():
    parser = ArgumentParser(description="Reconhecimento facial do desafio 2.")
    parser.add_argument("--dataset", type=Path, default=BASE_DIR / "dataset")
    parser.add_argument("--test-image", type=Path, default=BASE_DIR / "test" / "modelo_2_1.jpeg")
    parser.add_argument("--detector-dir", type=Path, default=BASE_DIR / "face_detection_model")
    parser.add_argument("--embedder", type=Path, default=BASE_DIR / "openface_nn4.small2.v1.t7")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "output")
    parser.add_argument(
        "--output-image",
        type=Path,
        default=BASE_DIR / "output" / "desafio_2_resultado.jpeg",
    )
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--resize-to", type=int, default=600)
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    face_detector, face_embedder = load_models(args.detector_dir, args.embedder)

    if not args.skip_train:
        train_recognizer(
            dataset_dir=args.dataset,
            output_dir=args.output_dir,
            detector=face_detector,
            embedder=face_embedder,
            min_confidence=args.min_confidence,
            resize_to=args.resize_to,
        )

    recognize_image(
        image_path=args.test_image,
        output_path=args.output_image,
        output_dir=args.output_dir,
        detector=face_detector,
        embedder=face_embedder,
        min_confidence=args.min_confidence,
        resize_to=args.resize_to,
    )
