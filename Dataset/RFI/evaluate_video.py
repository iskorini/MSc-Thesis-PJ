import argparse
import os
import sys

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption, draw_detections
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.losses import smooth_l1, focal
from keras_retinanet.preprocessing import csv_generator
from keras_retinanet.utils.eval import evaluate
from keras_retinanet.bin.debug import make_output_path
# other imports
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import progressbar
from rtree import index
from natsort import natsorted

def compute_intersection_over_union(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box_1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box_2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    intersection_over_union = intersection_area / float(box_1_area + box_2_area - intersection_area)
    return intersection_over_union

def calc_bbox_size(bbox):
    """
    Get the size of bbox in input
    # Arguments
        bbox       : coordinates of bbox (x_min, y_min, x_max, y_max).
    # Returns
        size of bbox
    """
    return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])

def iou_evaluation(detections, threshold):
    index_max_detections = np.argmax(list(map(lambda x: len(x), detections)))
    id_tree = 0
    idx = index.Index(interleaved=True)
    iou_values = [0]*len(detections[index_max_detections])
    for_debug = np.concatenate(np.delete(detections, index_max_detections, axis = 0))
    for detection in for_debug:
        idx.insert(id_tree, detection[0:4], obj = detection[4:6])
        id_tree = id_tree + 1
    for i in range(len(detections[index_max_detections])):
        intersection_bbox = idx.intersection(detections[index_max_detections][i][0:4], objects=True)
        num_elem = 0
        for item in intersection_bbox:
            num_elem = i+1
            iou_values[i] += compute_intersection_over_union(item.bbox, detections[index_max_detections][i][0:4])
        if num_elem is not 0:
            iou_values[i] /= num_elem
    iou_values = np.divide(iou_values, len(detections)-1)
    bbox = []
    labels = []
    scores = []
    for det, iou in zip(detections[index_max_detections], iou_values):
        if iou > threshold:
            intersection_bbox = idx.intersection(det[0:4], objects=True)
            gen = list(intersection_bbox)
            max_bbox_index = np.argmax(list(map(lambda x: calc_bbox_size(x), list(map(lambda x: x.bbox, gen)))))
            bbox.append(gen[max_bbox_index].bbox)
            scores.append(max(list(map(lambda x: x.object[0], gen))))
            labels.append(gen[max_bbox_index].object[1])
    return np.array(bbox), np.array(scores), np.array(labels, dtype=int), iou_values

def create_generator(annotations, classes):
    generator = csv_generator.CSVGenerator(
        annotations,
        classes,
        shuffle_groups=False,
        auto_augment=None,
        rand_augment=None,
        config=None
        )
    return generator

def get_all_detections(model, generator, path_list, score_threshold):
    all_detections_metrics = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]
    all_detections = []
    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        image    = generator.load_image(path_list[i][0])
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
        indices = np.where(scores[0, :] > score_threshold)[0]
        scores = scores[0][indices]
        scores_sort = np.argsort(-scores)[:100]
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        all_detections.append(image_detections)
        
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue
            all_detections_metrics[i][label] = image_detections[image_detections[:, -1] == label, :-1]
    return all_detections

def get_new_detections(all_detections, iou_treshold, iou_frames):
    new_detections = []
    for i in progressbar.progressbar(range(iou_frames+4, generator.size(), 1), prefix='Applying IoU over time: '):
        partial_detections = np.array(all_detections[i-iou_frames:i])
        frame_detection = iou_evaluation(partial_detections, iou_treshold)
        new_detections.append(frame_detection)
    return new_detections

def save_images(path_list, detections, generator, score_threshold, save_path, iou_frames):
    for i in progressbar.progressbar(range(iou_frames, generator.size(), 1), prefix='Saving images: '):
        #draw_annotations(generator.load, generator.load_annotations(i), label_to_name=generator.label_to_name)
        image = generator.load_image(path_list[i][0])
        draw_detections(
            image, 
            detections[i][0], 
            detections[i][1], 
            detections[i][2], 
            label_to_name=generator.label_to_name, score_threshold=score_threshold
            )
        output_path = make_output_path(save_path, generator.image_path(path_list[i][0]), flatten=False)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    setup_gpu(0)
    score_threshold = 0.3
    iou_frames = 3
    iou_treshold = 0.2
    dataset_path = Path('/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/RFI')
    annotations = dataset_path.joinpath('test_video/video1/dataset.csv')
    classes = dataset_path.joinpath('class.csv')
    generator = create_generator(annotations, classes)
    weights_path = Path('/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/RFI/frames/snapshots')
    save_path = dataset_path.joinpath('annotated_frames')
    model = models.load_model(weights_path.joinpath('resnet50_csv_14.h5'), backbone_name='resnet50')
    model = models.convert_model(model, anchor_params=None)
    path_list = []
    for i in range(generator.size()):
        path_list.append([i, Path(generator.image_path(i))])
    path_list = natsorted(path_list, key=lambda y: y[1])
    all_detections = get_all_detections(model, generator, path_list, score_threshold)
    new_detections = get_new_detections(all_detections, iou_treshold, iou_frames)
    save_images(path_list, new_detections, generator, score_threshold, save_path, iou_frames)
