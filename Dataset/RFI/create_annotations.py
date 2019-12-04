from pathlib import Path
import csv
import os

dataset_path = Path('/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/RFI/frames')

if __name__ == "__main__":
    with open('dataset.csv', 'w' ) as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for file in os.listdir(dataset_path):
            path = dataset_path.joinpath(file)
            filewriter.writerow([str(path), '', '', '', '', ''])