#Script for model evaluation in terms of various object detection metrics and confusion matrix
import os
import json
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from sklearn.metrics import confusion_matrix
import seaborn as sns
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score

img_dir = "dataset/images"
json_file = "dataset/annotations/instances_train.json"

def generate_confusion_matrix(predictor, dataset, num_classes):
    y_true = []
    y_pred = []
    
    for d in dataset:
        img = plt.imread(d["file_name"])
        
        true_classes = [ann["category_id"] for ann in d["annotations"]]
        
        outputs = predictor(img)
        pred_classes = outputs["instances"].pred_classes.cpu().numpy()
        pred_scores = outputs["instances"].scores.cpu().numpy()
        threshold = 0.5
        pred_classes = pred_classes[pred_scores > threshold]
        
        matched_true = []
        matched_pred = []
        
        for true_class in true_classes:
            if len(pred_classes) > 0:
                matched_true.append(true_class)
                matched_pred.append(pred_classes[0])
                pred_classes = pred_classes[1:]
            else:
                matched_true.append(true_class)
                matched_pred.append(num_classes)  
        
        for pred in pred_classes:
            matched_true.append(num_classes)  
            matched_pred.append(pred)
        
        y_true.extend(matched_true)
        y_pred.extend(matched_pred)
    
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes + 1))

    precision = precision_score(y_true, y_pred, average='weighted', labels=range(num_classes))
    recall = recall_score(y_true, y_pred, average='weighted', labels=range(num_classes))
    f1 = f1_score(y_true, y_pred, average='weighted', labels=range(num_classes))
    
    cm_normalized = normalize(cm, axis=1, norm='l1')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    class_names = MetadataCatalog.get("my_dataset_val").thing_classes + ['No Detection/FP']
    plt.xticks(np.arange(num_classes + 1) + 0.5, class_names, rotation=90)
    plt.yticks(np.arange(num_classes + 1) + 0.5, class_names, rotation=0)
    plt.tight_layout()
    plt.show()
    
    return cm_normalized, precision, recall, f1, y_true, y_pred

register_coco_instances("my_dataset_val", {}, json_file, img_dir)
MetadataCatalog.get("my_dataset_val").set(thing_classes=["car", "bike", "auto", "rickshaw", "cycle", "bus", "minitruck", "truck", "van", "taxi", "motorvan", "toto", "train", "boat", "cycle van"])

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.MODEL.WEIGHTS = "output/model_final.pth"  # Path to your model weights
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
cfg.MODEL.RETINANET.NUM_CLASSES = 15  # Adjust if necessary
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

val_loader = build_detection_test_loader(cfg, "my_dataset_val")

evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir="./output/")

print("Dataset information:")
dataset_dicts = DatasetCatalog.get("my_dataset_val")
print(f"Number of images in dataset: {len(dataset_dicts)}")

print("Running inference on validation dataset...")
results = inference_on_dataset(predictor.model, val_loader, evaluator)

print("\nCOCO Metrics:")
if "bbox" in results:
    bbox_results = results["bbox"]
    metrics = [
        "AP", "AP50", "AP75", "APs", "APm", "APl",
        "AR@1", "AR@10", "AR@100", "ARs@100", "ARm@100", "ARl@100"
    ]
    for metric in metrics:
        if metric in bbox_results:
            print(f"{metric}: {bbox_results[metric]:.4f}")
        else:
            print(f"{metric}: Not available")
else:
    print("No bbox results found")

dataset = DatasetCatalog.get("my_dataset_val")
num_classes = len(MetadataCatalog.get("my_dataset_val").thing_classes)
cm, precision, recall, f1, y_true, y_pred = generate_confusion_matrix(predictor, dataset, num_classes)

print("\nAdditional Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nNormalized Confusion Matrix:")
print(cm)

