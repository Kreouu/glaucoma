"""Fixed mean-area view matching the mild training crop, before letterbox."""
import math
from PIL import Image


def center_crop(image):
    fraction = math.sqrt(.9)
    width, height = image.size
    cropped_width = math.ceil(width * fraction)
    cropped_height = math.ceil(height * fraction)
    left, top = (width - cropped_width) // 2, (height - cropped_height) // 2
    return image.crop((left, top, left + cropped_width, top + cropped_height))


def preprocess_center(path, predictor):
    with Image.open(path) as image:
        if predictor.config['jpeg_draft'] and image.format == 'JPEG':
            image.draft('RGB', (predictor.config['size'], predictor.config['size']))
        return predictor.transform(center_crop(image.convert('RGB')))
