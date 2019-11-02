import pandas as pd 
from pathlib import Path
import csv

source_path = './Dataset/KAIST_MPD/fine_tuning_kaist_cars/random/set09/V000/via_export_csv.csv'
dataset_path = Path('/data/datasets/KAIST_MPD/rgbt-ped-detection/data/kaist-rgbt/images/set09/V000/lwir')

if __name__ == "__main__":
    df = pd.read_csv(source_path)
    with open('SET09.csv', 'w' ) as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for index, row in df.iterrows():
            path = dataset_path.joinpath(row['filename'])
            if row['region_count'] > 0:
                data = row['region_shape_attributes'].replace('{', '').replace('}','').replace('"', '').split(',')
                x = int(data[1].replace('x:', ''))
                y = int(data[2].replace('y:', ''))
                x2 = int(data[3].replace('width:', '')) + x
                y2 = int(data[4].replace('height:', '')) + y
                filewriter.writerow([str(path)]+[x, y, x2, y2]+['car'])
            else:
                filewriter.writerow([str(path), '',  '', '', '', ''])

