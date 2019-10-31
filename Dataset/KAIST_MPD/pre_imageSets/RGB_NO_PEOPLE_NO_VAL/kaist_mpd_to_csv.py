import glob
import csv
import itertools
import progressbar
import re
annotations_path = '/data/datasets/KAIST_MPD/rgbt-ped-detection/data/kaist-rgbt/annotations/**/**/**.txt'
#/data/datasets/KAIST_MPD/rgbt-ped-detection/data/kaist-rgbt/annotations/set00/V000/I02220.txt
#/data/datasets/KAIST_MPD/rgbt-ped-detection/data/kaist-rgbt/images/set00/V000/lwir/I0220.jpg
#/data/datasets/KAIST_MPD/rgbt-ped-detection/data/kaist-rgbt/annotations/**/**/**.txt
if __name__ == "__main__":
    tot = len(glob.glob(annotations_path))
    bar = progressbar.ProgressBar(maxval=tot, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    i = 0
    bar.start()
    with open('dataset_training.csv', 'w' ) as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for file_name in glob.iglob(annotations_path):
            i = i+1
            bar.update(i)
            data = open(file_name, 'r').read().split('\n')
            file_n = file_name.replace('/annotations/', '/images/').replace('.txt', '.jpg')
            file_n = file_n.replace(re.search('V...', file_n).group(0), re.search('V...', file_n).group(0)+'/lwir')
            if len(data)>2:
                for i in range(1, len(data)-1):
                    infos = data[1].split()
                    if infos[0].replace('?','') != 'people':
                        x1, y1 = infos[1], infos[2]
                        x2, y2 = str((int(infos[1])+int(infos[3]))-1), str((int(infos[2])+int(infos[4]))-1)
                        filewriter.writerow([file_n]+[x1, y1, x2, y2]+[infos[0].replace('?','')])
            elif len(data) == 2:
                filewriter.writerow([file_n, '',  '', '', '', ''])
    bar.finish()