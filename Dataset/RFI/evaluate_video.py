# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.visualization import draw_box, draw_caption, draw_detections
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.preprocessing import csv_generator
from keras_retinanet.utils.eval import _get_annotations, _compute_ap
from keras_retinanet.bin.debug import make_output_path
from keras_retinanet.utils.anchors import compute_overlap
# other imports
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import progressbar
from rtree import index
from natsort import natsorted
#credits: https://github.com/fizyr/keras-retinanet

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
    return np.array(bbox), np.array(scores), np.array(labels, dtype=int)

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
    all_detections_metrics = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size()-9400)]
    all_detections = []
    for i in progressbar.progressbar(range(generator.size()-9400), prefix='Running network: '):
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
    return all_detections, all_detections_metrics

def get_new_detections_1(all_detections, iou_threshold, iou_frames, generator):
    new_detections = []
    new_detections_metric = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size()-9400)]
    for i in range(iou_frames):
        new_detections.append([np.array(all_detections[i][:, 0:4]), np.array(all_detections[i][:, 4]), np.array(all_detections[i][:, 5], dtype=np.int)])
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue
            new_detections_metric[i][label] = all_detections[i][all_detections[i][:, -1] == label, :-1]
    for i in progressbar.progressbar(range(iou_frames, generator.size()-9400, 1), prefix='Applying IoU over time: '):
        partial_detections = np.array(all_detections[i-iou_frames:i])
        frame_detection = iou_evaluation(partial_detections, iou_threshold)
        new_detections.append(frame_detection)
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue
            if len(frame_detection[0])>0:
                a = np.concatenate([frame_detection[0], np.expand_dims(frame_detection[1], axis=1), np.expand_dims(frame_detection[2], axis=1)], axis=1)
                new_detections_metric[i][label] = a[a[:, -1] == label, :-1]
            else:
                a = np.array([])
                new_detections_metric[i][label] = a
    return new_detections, new_detections_metric

def get_new_detections_2(all_detections, iou_threshold, iou_frames, generator):
    new_detections = []
    new_detections_metric = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size()-9400)]
    for i in range(iou_frames):
        new_detections.append([np.array(all_detections[i][:, 0:4]), np.array(all_detections[i][:, 4]), np.array(all_detections[i][:, 5], dtype=np.int)])
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue
            new_detections_metric[i][label] = all_detections[i][all_detections[i][:, -1] == label, :-1]
    for i in progressbar.progressbar(range(iou_frames, generator.size()-9400, iou_frames), prefix='Applying IoU over time: '):
        partial_detections = np.array(all_detections[i-iou_frames:i])
        frame_detection = iou_evaluation(partial_detections, iou_threshold)
        for i in range(iou_frames):
            new_detections.append(frame_detection)
            for label in range(generator.num_classes()):
                if not generator.has_label(label):
                    continue
                if len(frame_detection[0])>0:
                    a = np.concatenate([frame_detection[0], np.expand_dims(frame_detection[1], axis=1), np.expand_dims(frame_detection[2], axis=1)], axis=1)
                    new_detections_metric[i][label] = a[a[:, -1] == label, :-1]
                else:
                    a = np.array([])
                    new_detections_metric[i][label] = a
                
    return new_detections, new_detections_metric


def save_images(path_list, detections, generator, score_threshold, save_path, iou_frames):
    for i in progressbar.progressbar(range(generator.size()-9400), prefix='Saving images: '):
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

def evaluate(generator, detections, iou_threshold):
    all_detections     = detections
    all_annotations    = _get_annotations(generator)
    average_precisions = {}
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()-9400):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions

def main():
    setup_gpu(1)
    score_threshold = 0.3
    iou_frames = 5
    iou_ot_threshold = 0.1
    iou_threshold = 0.5
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
    all_detections, all_detections_metrics = get_all_detections(model, generator, path_list, score_threshold)
    new_detections, new_detections_metrics = get_new_detections_2(all_detections, iou_ot_threshold, iou_frames, generator)
    print(len(new_detections))
    print(new_detections[7])
    print('----------')
    print(new_detections[0])
    save_images(path_list, new_detections, generator, score_threshold, save_path, iou_frames)

if __name__ == "__main__":
    main()
