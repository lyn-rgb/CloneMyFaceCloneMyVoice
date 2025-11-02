from typing import Union, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import torch


def resize_image_to_bucket(image: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]) -> np.ndarray:
    """
    Resize the image to the bucket resolution.

    bucket_reso: **(width, height)**
    """
    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image

    bucket_width, bucket_height = bucket_reso

    # resize the image to the bucket resolution to match the short side
    scale_width = bucket_width / image_width
    scale_height = bucket_height / image_height
    scale = max(scale_width, scale_height)
    image_width = int(image_width * scale + 0.5)
    image_height = int(image_height * scale + 0.5)

    if scale > 1:
        image = Image.fromarray(image) if not is_pil_image else image
        image = image.resize((image_width, image_height), Image.LANCZOS)
        image = np.array(image)
    else:
        image = np.array(image) if is_pil_image else image
        image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)

    # crop the image to the bucket resolution
    crop_left = (image_width - bucket_width) // 2
    crop_top = (image_height - bucket_height) // 2
    image = image[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]
    return image

# prepare image
def preprocess_image(image: Image, w: int, h: int) -> Tuple[torch.Tensor, np.ndarray, Optional[np.ndarray]]:
    """
    Preprocess the image for the model.
    Args:
        image (Image): The input image. RGB or RGBA format.
        w (int): The target bucket width.
        h (int): The target bucket height.
    Returns:
        Tuple[torch.Tensor, np.ndarray, Optional[np.ndarray]]:
            - image_tensor: The preprocessed image tensor (NCHW format). -1.0 to 1.0.
            - image_np: The original image as a numpy array (HWC format). 0 to 255.
            - alpha: The alpha channel if present, otherwise None.
    """
    if image.mode == "RGBA":
        alpha = image.split()[-1]
    else:
        alpha = None
    image = image.convert("RGB")

    image_np = np.array(image)  # PIL to numpy, HWC

    image_np = resize_image_to_bucket(image_np, (w, h))
    image_tensor = torch.from_numpy(image_np).float() / 127.5 - 1.0  # -1 to 1.0, HWC
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW -> NCHW, N=1
    return image_tensor, image_np, alpha
