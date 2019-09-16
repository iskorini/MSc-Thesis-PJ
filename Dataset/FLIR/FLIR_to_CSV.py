import json
import glob
import csv
import itertools
import progressbar
#/data/datasets/FLIR_ADAS/FLIR_ADAS/training/Annotations
#/data/datasets/FLIR_ADAS/FLIR_ADAS/validation/Annotations
if __name__ == "__main__":
    annotations_path = '/data/datasets/FLIR_ADAS/FLIR_ADAS/validation/Annotations/*.json'
    id_to_name = {
        '1': 'People',
        '2': 'Bicycles',
        '3': 'Cars',
        '18': 'Dogs',
        '91': 'Other'
    }
    tot = len(glob.glob(annotations_path))
    bar = progressbar.ProgressBar(maxval=tot, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    i = 0
    bar.start()
    with open('dataset_RGB_validation.csv', 'w' ) as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for file_name in glob.iglob(annotations_path):
            data = json.load(open(file_name))
            if len(data['annotation'])>0:
                for i in range(len(data['annotation'])):
                    x1, y1 = data['annotation'][i]['segmentation'][0][0],data['annotation'][i]['segmentation'][0][1]
                    x2, y2 = data['annotation'][i]['segmentation'][0][4],data['annotation'][i]['segmentation'][0][5]
                    filewriter.writerow([file_name.replace('Annotations', 'RGB')]+[x1, y1, x2, y2]+[id_to_name[data['annotation'][i]['category_id']]])
            else:
                filewriter.writerow([file_name, '',  '', '', '', ''])
            i = i+1
            bar.update(i)
    bar.finish()