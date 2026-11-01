"""Mild random training crop without square-crop fallbacks on wide images."""
import math
import random


class RandomRelativeCrop:
    def __init__(self, min_area):
        self.min_area = min_area

    def __call__(self, image):
        if self.min_area == 1:
            return image
        fraction = math.sqrt(random.uniform(self.min_area, 1))
        width, height = image.size
        cropped_width = math.ceil(width * fraction)
        cropped_height = math.ceil(height * fraction)
        left = random.randint(0, width - cropped_width)
        top = random.randint(0, height - cropped_height)
        return image.crop((left, top, left + cropped_width, top + cropped_height))
