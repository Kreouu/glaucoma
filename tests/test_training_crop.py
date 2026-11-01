import random
import unittest
from unittest.mock import patch
from PIL import Image
from training_crop import RandomRelativeCrop


class CropTest(unittest.TestCase):
    def test_disabled_is_identity_and_preserves_rng(self):
        image=Image.new('RGB',(200,100));state=random.getstate()
        self.assertIs(RandomRelativeCrop(1)(image),image);self.assertEqual(random.getstate(),state)

    def test_wide_image_retains_fraction_and_correct_pixels(self):
        image=Image.new('RGB',(200,100));image.putpixel((20,10),(255,30,10))
        with patch('training_crop.random.uniform',return_value=.81),patch('training_crop.random.randint',side_effect=[20,10]):
            result=RandomRelativeCrop(.8)(image)
        self.assertEqual(result.size,(180,90));self.assertEqual(result.getpixel((0,0)),(255,30,10))
        self.assertEqual(result.width/result.height,image.width/image.height)
        random.seed(42)
        for size in [(1,1),(3,5),(200,100),(100,200)]:
            for _ in range(50):
                result=RandomRelativeCrop(.8)(Image.new('RGB',size))
                self.assertGreaterEqual(result.width*result.height/(size[0]*size[1]),.8)
                self.assertLessEqual(result.width,size[0]);self.assertLessEqual(result.height,size[1])


if __name__=='__main__':unittest.main()
