from pathlib import Path
from struct import unpack

import numpy as np

import cv2


# The flip = 1 means that the metric will multiply 1,
# which means unchanged. So flip = -1 will flip the image.
def read_image(path: Path):
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_metrics(path: Path):
    with path.open('rb') as f:
        return unpack('f' * 12, f.read())
