#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

#import comet_ml
from comet_ml import Experiment
from comet_ml import Optimizer
API_KEY = 'spsZWUefep11cUugBplS6AKQm' #os.environ['COMET_API_KEY']

# import keras
import keras
from keras import backend as K 

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.losses import smooth_l1, focal
from keras_retinanet.preprocessing import csv_generator
from keras_retinanet.utils.eval import evaluate

# other imports
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

setup_gpu(1)


# In[2]:


default_params = {
    'N': 3,
    'M': 10
}
dataset_path = Path('/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/KAIST_MPD/fine_tuning_kaist_cars/ds')
train_annotations = dataset_path.joinpath('train_no_people.csv')
test_annotations = dataset_path.joinpath('test_w_people.csv')
classes = Path('/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/KAIST_MPD/class_name_to_ID_CARS.csv')

def create_train_generator(N, M):
    train_generator = csv_generator.CSVGenerator(
        train_annotations,
        classes,
        transform_generator=None,
        visual_effect_generator=None,
        image_min_side=800,
        image_max_side=1333,
        auto_augment=None,
        rand_augment=(N,M),
        config=None
        )
    return train_generator

def create_test_generator():
    test_generator = csv_generator.CSVGenerator(
    test_annotations,
    classes,
    shuffle_groups=False,
    auto_augment=None,
    rand_augment=None,
    config=None)
    return test_generator
    
train_generator = create_train_generator(default_params['N'], default_params['M'])
test_generator = create_test_generator()


# In[3]:


def create_model():
    model_path = Path('/data/students_home/fschipani/thesis/keras-retinanet/snapshots/resnet50_coco_best_v2.1.0.h5')
    model = models.backbone('resnet50').retinanet(num_classes=train_generator.num_classes())
    model.load_weights(model_path, by_name=True, skip_mismatch=True)
    model.compile(
        loss={
            'regression'    : smooth_l1(),
            'classification': focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )
    return model


# In[4]:


def test_model(model, generator, score_threshold):
    result = evaluate(
        generator,
        models.convert_model(model, None),
        score_threshold=score_threshold,
        iou_threshold=0.5,
        max_detections=100,
        save_path=None
    )
    return result

test_results = test_model(create_model(), create_test_generator(), 0.3)


# In[5]:


test_results[2][0] #mAP for person


# In[6]:


config = {
      "algorithm": "bayes",
      "parameters": {
          "N": {"type": "integer", "min": 1, "max": 3},
          "M": {"type": "integer", "min": 5, "max": 30},
      },
      "spec": {
          "metric": "mAP",
          "objective": "minimize",
      },
  }
opt = Optimizer(config, api_key=API_KEY, project_name="MSC-Thesis-PJ")


# In[ ]:


for experiment in opt.get_experiments():
    N = experiment.get_parameter('N')
    M = experiment.get_parameter('M')
    train_generator = create_train_generator(N, M)
    model = create_model()
    history = model.fit_generator(generator=train_generator, steps_per_epoch=10000, epochs=5, verbose=1)
    test_results = test_model(model, create_test_generator(), 0.3)
    experiment.log_metric("mAP", -test_results[2][0])
    K.clear_session()
    

