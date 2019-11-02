from pathlib import Path
import glob
from skimage import io, img_as_float
import numpy as np
from multiprocessing.pool import Pool as ThreadPool

images_path = Path('/data/datasets/KAIST_MPD/rgbt-ped-detection/data/kaist-rgbt/images/')

day_train = ['set00', 'set01', 'set02']
day_test = ['set06', 'set07', 'set08']
night_train = ['set03', 'set04', 'set05']
night_test = ['set09', 'set10', 'set11']


def mean_calc(image_set, name_set):
    values = []
    for set in image_set:
        for img_pth in glob.iglob(str(images_path.joinpath(set)) + '/**/lwir/**.jpg'):
            image_matrix = img_as_float(io.imread(img_pth, as_gray=True))
            values = np.append(values, np.mean(image_matrix))
    print(name_set +" mean: " +str(np.mean(values)))


if __name__ == '__main__':
    pool = ThreadPool(4)
    args1 = [day_train, day_test, night_train, night_test]
    args2 = ['day_train', 'day_test', 'night_train', 'night_test']
    results = pool.starmap(mean_calc, zip(args1, args2))
    pool.close()
    pool.join()

