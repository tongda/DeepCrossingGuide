from pathlib import Path
from struct import unpack

import cv2


def read_image(path: Path):
    image = cv2.imread(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_metrics(path: Path):
    with path.open('rb') as f:
        return unpack('f' * 12, f.read())
