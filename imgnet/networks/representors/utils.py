import os
from pathlib import Path

import numpy as np
from PIL import Image


MODEL_ZOO = Path(os.environ.get("DM_MODEL_ZOO", "/path/to/your/model_zoo"))


def pretrained_path(*parts):
    return str(MODEL_ZOO / "pretrained" / Path(*parts))


def require_path(path, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def center_crop_arr(pil_image, image_size):
    """Resize and center-crop using OpenAI guided-diffusion's ADM helper.

    Source: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )
