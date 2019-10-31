import os
import random
from pathlib import Path
import shutil

source = Path('./images/set09/v000/annotations')
destination = Path('./random/set09/v000/annotations')

if __name__ == "__main__":
    for file in random.sample(os.listdir(source), 200):
        annotation_path = source.joinpath(file)
        lwir_path = source.parent.joinpath('lwir/'+file.replace('.txt', '.jpg'))
        visible_path = source.parent.joinpath('visible/'+file.replace('.txt', '.jpg'))
        shutil.copyfile(annotation_path, Path(str(annotation_path).replace('images', 'random')))
        shutil.copyfile(lwir_path, Path(str(lwir_path).replace('images', 'random')))
        shutil.copyfile(visible_path, Path(str(visible_path).replace('images', 'random')))