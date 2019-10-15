from keras_retinanet.utils.transform import random_transform
from keras_retinanet.utils.image import apply_transform, TransformParameters, adjust_transform_for_image, read_image_bgr
from matplotlib.image import imsave

if __name__ == "__main__":
    img = read_image_bgr('./TestFile/foo.jpg')
    transform = random_transform(flip_x_chance=1, flip_y_chance=0)
    print(transform)
    transform_parameters = TransformParameters()
    transform = adjust_transform_for_image(transform, img, transform_parameters.relative_translation)
    img_transformed = apply_transform(transform, img, transform_parameters)
    imsave('./TestFile/foo_transformed.jpg', img_transformed)
    exit(0)