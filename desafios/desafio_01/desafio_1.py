from argparse import ArgumentParser
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent


def aplicar_desfoque_fundo(
    image_path: Path,
    cascade_path: Path,
    output_path: Path,
    padding: int = 100,
    blur_kernel: int = 15,
    blur_sigma: int = 20,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel carregar a imagem: {image_path}")

    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise FileNotFoundError(f"Nao foi possivel carregar o classificador: {cascade_path}")

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=3)
    if len(faces) == 0:
        raise RuntimeError("Nenhuma face foi detectada na imagem.")

    if blur_kernel % 2 == 0:
        blur_kernel += 1

    final_image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), blur_sigma)

    for x, y, width, height in faces:
        y1 = max(0, y - padding)
        y2 = min(y + height + padding, image.shape[0])
        x1 = max(0, x - padding)
        x2 = min(x + width + padding, image.shape[1])

        final_image[y1:y2, x1:x2, :] = image[y1:y2, x1:x2, :]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), final_image):
        raise RuntimeError(f"Nao foi possivel salvar a imagem em: {output_path}")

    print(f"Imagem salva em: {output_path}")
    print(f"Faces detectadas: {len(faces)}")


def parse_args():
    parser = ArgumentParser(
        description="Aplica desfoque no fundo da imagem mantendo a regiao da face nitida."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=BASE_DIR / "desafio_1.jpeg",
        help="Caminho da imagem de entrada.",
    )
    parser.add_argument(
        "--cascade",
        type=Path,
        default=BASE_DIR / "haarcascade_frontalface_default.xml",
        help="Caminho do classificador Haar Cascade.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "desafio_1_resultado.jpeg",
        help="Caminho da imagem de saida.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=100,
        help="Padding em pixels ao redor da face detectada.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=15,
        help="Tamanho do kernel do GaussianBlur. Se for par, sera ajustado para impar.",
    )
    parser.add_argument(
        "--blur-sigma",
        type=int,
        default=20,
        help="Sigma usado no GaussianBlur.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    aplicar_desfoque_fundo(
        image_path=args.image,
        cascade_path=args.cascade,
        output_path=args.output,
        padding=args.padding,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
    )
