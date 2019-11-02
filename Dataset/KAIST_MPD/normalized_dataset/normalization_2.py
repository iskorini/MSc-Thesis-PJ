from pathlib import Path
import glob
from skimage import io, img_as_float
import numpy as np
from multiprocessing.pool import Pool as ThreadPool
import os
import progressbar

images_path = Path('/data/datasets/KAIST_MPD/rgbt-ped-detection/data/kaist-rgbt/images/')

day_train = ['set00', 'set01', 'set02']
day_test = ['set06', 'set07', 'set08']
night_train = ['set03', 'set04', 'set05']
night_test = ['set09', 'set10', 'set11']


def mean_calc(image_set, name_set):
    means = []
    total = 0
    for set in image_set:
        i = 0
        values = []
        tot = len(glob.glob(str(images_path.joinpath(set)) + '/**/lwir/**.jpg'))
        bar = progressbar.ProgressBar(maxval=tot,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for img_pth in glob.iglob(str(images_path.joinpath(set)) + '/**/lwir/**.jpg'):
            i = i+1
            values.append(img_as_float(io.imread(img_pth, as_gray=True)))
            bar.update(i)
        bar.finish()
        means.append((np.mean(list(map(np.ndarray.flatten, values)), axis=0)) * i )
        total = total + i
    mean = sum(means)/total
    save_pth = Path(os.getcwd()+'/mean_matrix/'+name_set)
    np.save(save_pth, mean)
    print(name_set +" mean: done -> "+str(save_pth))


if __name__ == '__main__':
    mean_calc(day_train, 'day_train')
    #pool = ThreadPool(4)
    #args1 = [day_train, day_test, night_train, night_test]
    #args2 = ['day_train', 'day_test', 'night_train', 'night_test']
    #results = pool.starmap(mean_calc, zip(args1, args2))
    #pool.close()
    #pool.join()

