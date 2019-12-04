import os

import pandas as pd
import csv
from pathlib import Path
import glob
import random

path_frames_0 = Path('/Users/fschipani/Desktop/Tesi/MSc-Thesis-PJ/Dataset/RFI/frames/ds/frames_0')
path_frames_1 = Path('/Users/fschipani/Desktop/Tesi/MSc-Thesis-PJ/Dataset/RFI/frames/ds/frames_1')
path_frames_2 = Path('/Users/fschipani/Desktop/Tesi/MSc-Thesis-PJ/Dataset/RFI/frames/ds/frames_2')

if __name__ == "__main__":
    all_images_frames_0 = os.listdir(path_frames_0)
    test_set_frames_0 = random.sample(all_images_frames_0, int(len(all_images_frames_0)/4))
    train_set_frames_0 = list(set(all_images_frames_0)-set(test_set_frames_0))
    annotations = pd.read_csv('ds/frames_1_csv.csv')
    all_images_frames_1 = os.listdir(path_frames_1)
    test_set_frames_1 = random.sample(all_images_frames_1, int(len(all_images_frames_1)/4))
    train_set_frames_1 = list(set(all_images_frames_1)-set(test_set_frames_1))
    all_images_frames_2 = os.listdir(path_frames_2)
    test_set_frames_2 = random.sample(all_images_frames_1, int(len(all_images_frames_2)/4))
    train_set_frames_2 = list(set(all_images_frames_2)-set(test_set_frames_2))
    with open('train_set.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for file in train_set_frames_0:
            image_path = path_frames_0.joinpath(file)
            filewriter.writerow([image_path, '', '', '', '', ''])
        for file in train_set_frames_1:
            image_path = path_frames_1.joinpath(file)
            for index, row in annotations.loc[annotations['filename'] == file].iterrows():
                if row['region_count'] == 0:
                    filewriter.writerow([image_path, '', '', '', '', ''])
                else:
                    data = row['region_shape_attributes'].replace('{', '').replace('}', '').replace('"', '').split(
                                ',')
                    x = int(data[1].replace('x:', ''))
                    y = int(data[2].replace('y:', ''))
                    x2 = int(data[3].replace('width:', '')) + x
                    y2 = int(data[4].replace('height:', '')) + y
                    filewriter.writerow([image_path] + [x, y, x2, y2] + ['person'])
        for file in train_set_frames_2:
            image_path = path_frames_2.joinpath(file)
            filewriter.writerow([image_path, '', '', '', '', ''])
    
    with open('test_set.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for file in test_set_frames_0:
            image_path = path_frames_0.joinpath(file)
            filewriter.writerow([image_path, '', '', '', '', ''])
        for file in test_set_frames_1:
            image_path = path_frames_1.joinpath(file)
            for index, row in annotations.loc[annotations['filename'] == file].iterrows():
                if row['region_count'] == 0:
                    filewriter.writerow([image_path, '', '', '', '', ''])
                else:
                    data = row['region_shape_attributes'].replace('{', '').replace('}', '').replace('"', '').split(
                                ',')
                    x = int(data[1].replace('x:', ''))
                    y = int(data[2].replace('y:', ''))
                    x2 = int(data[3].replace('width:', '')) + x
                    y2 = int(data[4].replace('height:', '')) + y
                    filewriter.writerow([image_path] + [x, y, x2, y2] + ['person'])
        for file in test_set_frames_2:
            image_path = path_frames_2.joinpath(file)
            filewriter.writerow([image_path, '', '', '', '', ''])

    #with open('frames_2.csv', 'w') as csvfile:
    #    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
