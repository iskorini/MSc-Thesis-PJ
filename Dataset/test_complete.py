"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
	http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from pathlib import Path
import glob 
import pandas as pd
# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import models
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters
from keras_retinanet.utils.eval import evaluate
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.tf_version import check_tf_version
from keras import backend as K

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', palette='Set2')

def create_generator(annotations, classes):
	""" Create generators for evaluation.
	"""
	validation_generator = CSVGenerator(
			annotations,
			classes,
			image_min_side=800,
			image_max_side=1333,
			config=None,
			shuffle_groups=False,
		)
	return validation_generator


def main():
	ax = plt.gca()
	#ax.set_aspect(20)

	score_threshold = [0.3]#, 0.4, 0.5, 0.6, 0.7, 0.8]
	setup_gpu(0)
	classes = Path('/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/KAIST_MPD/class_name_to_ID_CARS.csv')
	annotations = Path('/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/KAIST_MPD/fine_tuning_kaist_cars/ds/test_w_people.csv')
	save_path = Path('/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/tests_manual_annotations_08_cars/')
	generator = create_generator(annotations, classes)
	#weight = '/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/weights/manual_annotation_08.h5'
	weight_path_ra = '/data/students_home/fschipani/rand_augment/snapshots_ra_scratch'
	weight_path = '/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/snapshots/'

	# optionally load anchor parameters
	anchor_params = None
	dataframe = pd.DataFrame(columns=['epoch', 'score_threshold',
	  'person_instances', 'cyclist_instances', 'cars_instances', 
	  'map_person', 'map_cyclist', 'map_cars', 'weighted_map', 
	  'map', 'false_positives', 'true_positives', 'recall', 'precision'])
	for weight in glob.iglob(weight_path+'/*.h5'):
		print(weight)
		for threshold in [0.3]:
			os.makedirs(save_path.joinpath(str(threshold*100)), exist_ok=True)
			name_folder = weight.replace('/weights', '').replace('.h5', '')
			print(weight)    
			model = models.load_model(weight, backbone_name='resnet50')
			model = models.convert_model(model, anchor_params=anchor_params)
			os.makedirs(save_path.joinpath(str(threshold*100)).joinpath(name_folder), exist_ok=True)
			average_precisions, other_metrics = evaluate(
				generator,
				model,
				iou_threshold=0.5,
				score_threshold=threshold,
				max_detections=100,
				save_path=save_path.joinpath(str(threshold*100)).joinpath(name_folder)
			)
			total_instances = []
			precisions = []
			for label, (average_precision, num_annotations) in average_precisions.items():
				print('{:.0f} instances of class'.format(num_annotations),
					  generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
				total_instances.append(num_annotations)
				precisions.append(average_precision)
			if sum(total_instances) == 0:
				print('No test instances found.')
				return
			values = {
				'epoch': int(weight.replace('resnet50_csv_', '').replace('.h5', '').replace('/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/snapshots/', '')),
				'score_threshold': threshold,
				'person_instances':int(average_precisions[0][1]),
				'cyclist_instances':int(average_precisions[1][1]),
				'cars_instances':int(average_precisions[2][1]),
				'map_person':average_precisions[0][0],
				'map_cyclist':average_precisions[1][0],
				'map_cars':average_precisions[2][0],
				'weighted_map': (sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)) ,
				'map': (sum(precisions) / sum(x > 0 for x in total_instances)),
				'false_positives':pd.Series(other_metrics[0]),
				'true_positives':pd.Series(other_metrics[1]),
				'recall':pd.Series(other_metrics[2]),
				'precision':pd.Series(other_metrics[3])
			}
			dataframe = dataframe.append(values, ignore_index=True)
			K.clear_session()
			#plt.plot(other_metrics[2][2], other_metrics[3][2], label = str(threshold)) #[recall, precision][class] usually class: 0->person 1->cyclist 2->cars
	dataframe.to_csv('./spero_sia_l_ultimo.csv')
	#plt.legend(ncol=1, loc='lower left')
	#ax.yaxis.grid(True) 
	#ax.xaxis.grid(False)
	#ax.set_xlabel('Recall')
	#ax.set_ylabel('Precision')
	#ax.margins()
	#ax.set_ylim(bottom=0)
	#plt.locator_params(axis='y', nbins=10)
	#plt.locator_params(axis='x', nbins=10)
	#sns.despine()
	##plt.show()
	#plt.savefig('./graphics_manual_annotations_08_cars.pdf', format='pdf', bbox_inches='tight')
if __name__ == '__main__':
	main()
