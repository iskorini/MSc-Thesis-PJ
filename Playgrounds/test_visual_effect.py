from keras_retinanet.utils.transform import random_transform, DEFAULT_PRNG
from keras_retinanet.utils.image import VisualEffect, read_image_bgr
from matplotlib.image import imsave
from math import pi
import numpy as np

if __name__ == "__main__":
    for i in range(0, 10):
        img = read_image_bgr('./TestFile/lwir.png')
        val_range = [0, 1]
        visual_effect = VisualEffect(
            contrast_factor= i/10,
            brightness_delta= i/10,
            hue_delta= 0,
            saturation_factor= 0
        )
        img_transformed = visual_effect(img)
        imsave('./TestFile/'+str(i)+'.jpg', img_transformed)
    exit(0)