from keras_retinanet.utils.transform import random_transform, DEFAULT_PRNG
from keras_retinanet.utils.image import apply_transform, TransformParameters, adjust_transform_for_image, read_image_bgr
from matplotlib.image import imsave
from math import pi

if __name__ == "__main__":
    for i in range(0, 5):
        img = read_image_bgr('./TestFile/foo.jpg')
        print(i)
        transform = random_transform(flip_x_chance=0, min_rotation=(i*pi)/2, max_rotation=(i*pi)/2, prng=DEFAULT_PRNG)
        print(transform)
        transform_parameters = TransformParameters()
        transform = adjust_transform_for_image(transform, img, transform_parameters.relative_translation)
        img_transformed = apply_transform(transform, img, transform_parameters)
        imsave('./TestFile/'+str(i)+'.jpg', img_transformed)
    exit(0)