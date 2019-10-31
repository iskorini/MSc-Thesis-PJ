import glob
import csv
import re
import os
from pathlib import Path
import argparse

annotations_path = Path('/data/datasets/KAIST_MPD/rgbt-ped-detection/data/kaist-rgbt/annotations/')
image_path = Path('/data/datasets/KAIST_MPD/rgbt-ped-detection/data/kaist-rgbt/images/')
image_types = ['visible', 'lwir']

if __name__ == "__main__":
    for file_name in glob.iglob(os.getcwd()+'/**.txt'):
        for image_type in image_types:
            csv_path_file = Path(file_name).cwd()/'csv_files'/image_type/(Path(file_name).stem+'.csv')
            with open(csv_path_file, 'w' ) as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                print(file_name +  ' -> ' + str(csv_path_file))
                file = open(file_name, 'r').read().split()
                for annotation in file:
                    ann = open(annotations_path / (annotation+'.txt'), 'r').read().split('\n')
                    file_n = (image_path/annotation).parts[0:-1]+(image_type,Path(annotation).stem+'.jpg')
                    file_n = os.path.join(*file_n)
                    if len(ann)>2:
                        for i in range(1, len(ann)-1):
                            infos = ann[i].split()
                            #if infos[0].replace('?','') != 'people':
                            x1, y1 = infos[1], infos[2]
                            x2, y2 = str((int(infos[1])+int(infos[3]))-1), str((int(infos[2])+int(infos[4]))-1)
                            filewriter.writerow([file_n]+[x1, y1, x2, y2]+[infos[0].replace('?','')])
                    elif len(ann) == 2:
                        filewriter.writerow([file_n, '',  '', '', '', ''])

            

