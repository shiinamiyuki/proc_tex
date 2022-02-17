import numpy as np
import cv2 as cv


def image(path: str, encoding='srgb', gamma=2.2):
    if encoding not in ['srgb', 'gamma', 'linear']:
        raise RuntimeError("encoding must be one of 'srgb','gamma','linear'")
    ext = path.split('.')[-1]

    img = cv.imread(path, cv.IMREAD_COLOR, dtype=np.float32)
    # if ext == 'exr':
    #     img = img.astype(np.float32)
    # elif ext in ['png','jpg','jpeg']:
    #     img = img.astype(np.float32) / 255.0
    # else:
    #     raise RuntimeError('unsupported format')
    if encoding == 'gamma':
        img = img ** (gamma)
    elif encoding == 'srgb':
        mask = img <= 0.04045
        img = np.select([mask, ~mask], [img / 12.92,
                        ((img + 0.055) / 1.055) ** 2.4])
    return img


# def 
