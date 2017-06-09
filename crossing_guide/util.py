from pathlib import Path
from struct import unpack

import scipy.ndimage as ndi
import numpy as np


def read_image(path: Path, flip=False):
    image = ndi.imread(str(path))
    if flip:
        image = np.fliplr(image)
    return image


def read_metrics(path: Path):
    with path.open('rb') as f:
        return unpack('f' * 12, f.read())
