from argparse import ArgumentParser
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent


def import_yolo():
    try:
        import torch
        import ultralytics
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics/torch nao estao instalados. Rode: pip install ultralytics torch"
        ) from exc

    return ultralytics, torch


def detect_pose(
    image_path: Path,
    model_path: Path,
    output_path: Path,
    conf: float = 0.25,
) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Imagem nao encontrada: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo YOLO de pose nao encontrado: {model_path}")

    ultralytics, torch = import_yolo()
    model = ultralytics.YOLO(str(model_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    results = model(str(image_path), conf=conf, verbose=False)
    if not results:
        raise RuntimeError("Nenhum resultado foi retornado pelo modelo.")

    plotted_image = results[0].plot()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), plotted_image):
        raise RuntimeError(f"Nao foi possivel salvar a imagem em: {output_path}")

    people_count = 0
    if results[0].keypoints is not None:
        people_count = len(results[0].keypoints)

    print(f"Imagem salva em: {output_path}")
    print(f"Pessoas com pose detectada: {people_count}")


def parse_args():
    parser = ArgumentParser(description="Reconhecimento de poses com YOLO.")
    parser.add_argument("--image", type=Path, default=BASE_DIR / "desafio_4.jpeg")
    parser.add_argument("--model", type=Path, default=BASE_DIR / "yolo11n-pose.pt")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "desafio_4_resultado.jpeg")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detect_pose(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        conf=args.conf,
    )
