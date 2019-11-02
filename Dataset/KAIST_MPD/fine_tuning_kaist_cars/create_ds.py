import os

import pandas as pd
import csv
from pathlib import Path
import glob

if __name__ == '__main__':
    sets = ['set05/V000', 'set00/V000']
    with open('./ds/train_no_people.csv', 'w' ) as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for set in sets:
            dataset_path = Path('/data/datasets/KAIST_MPD/rgbt-ped-detection/data/kaist-rgbt/images/'+set+'/lwir')
            cars_annotation = pd.read_csv('./random/' + set + '/via_export_csv.csv')
            for img in glob.iglob(os.getcwd()+'/random/' + set + '/annotations/**.txt'):
                df = cars_annotation.loc[cars_annotation['filename'] == str(Path(img).stem) + '.jpg']
                path = dataset_path.joinpath(df.iloc[0]['filename'])
                ann = open(img).read().split('\n')
                if len(ann) > 2:
                    for i in range(1, len(ann) - 1):
                        infos = ann[i].split()
                        if infos[0].replace('?','') != 'people':
                            x1, y1 = infos[1], infos[2]
                            x2, y2 = str((int(infos[1]) + int(infos[3])) - 1), str((int(infos[2]) + int(infos[4])) - 1)
                            filewriter.writerow([str(path)] + [x1, y1, x2, y2] + [infos[0].replace('?', '')])
                elif len(ann) == 2:
                    filewriter.writerow([str(path), '', '', '', '', ''])
                for index, row in df.iterrows():
                    a = a+1
                    if row['region_count'] > 0:
                        data = row['region_shape_attributes'].replace('{', '').replace('}', '').replace('"', '').split(
                            ',')
                        x = int(data[1].replace('x:', ''))
                        y = int(data[2].replace('y:', ''))
                        x2 = int(data[3].replace('width:', '')) + x
                        y2 = int(data[4].replace('height:', '')) + y
                        filewriter.writerow([str(path)] + [x, y, x2, y2] + ['cars'])
                    else:
                        filewriter.writerow([str(path), '', '', '', '', ''])


