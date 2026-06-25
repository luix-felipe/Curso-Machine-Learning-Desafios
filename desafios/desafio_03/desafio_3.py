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


def track_objects(
    source: Path,
    model_path: Path,
    output_path: Path,
    conf: float = 0.1,
    iou: float = 0.7,
) -> None:
    if not source.exists():
        raise FileNotFoundError(
            f"Fonte nao encontrada: {source}\n"
            "Adicione um video/imagem na pasta do desafio ou informe --source."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo YOLO nao encontrado: {model_path}")

    ultralytics, torch = import_yolo()
    model = ultralytics.YOLO(str(model_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    results = model.track(
        source=str(source),
        conf=conf,
        iou=iou,
        persist=True,
        stream=True,
        verbose=False,
    )

    writer = None
    frames_written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for result in results:
        frame = result.plot()
        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"XVID"),
                20,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Nao foi possivel criar o video: {output_path}")

        writer.write(frame)
        frames_written += 1

    if writer is not None:
        writer.release()

    if frames_written == 0:
        raise RuntimeError("Nenhum frame foi processado.")

    print(f"Video salvo em: {output_path}")
    print(f"Frames processados: {frames_written}")


def parse_args():
    parser = ArgumentParser(description="Rastreamento de objetos com YOLO.")
    parser.add_argument("--source", type=Path, default=BASE_DIR / "video.mov")
    parser.add_argument("--model", type=Path, default=BASE_DIR / "yolo11n.pt")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "output.avi")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    track_objects(
        source=args.source,
        model_path=args.model,
        output_path=args.output,
        conf=args.conf,
        iou=args.iou,
    )
