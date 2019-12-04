import os
import random
from pathlib import Path
import shutil

frames = [Path('./frames_0'), Path('frames_1'), Path('frames_2')]
destination = [Path('ds').joinpath(frames[0]), Path('ds').joinpath(frames[1]), Path('ds').joinpath(frames[2])]
numbers = [(0, 200), (1, 400), (2, 100)] 

if __name__ == "__main__":
    for index, number in numbers:
        for file in random.sample(os.listdir(frames[index]), number):
            shutil.copyfile(frames[index].joinpath(file), destination[index].joinpath(file))
