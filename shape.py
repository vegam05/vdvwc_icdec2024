#Script for checking input shape for debugging purposes only
import detectron2
from detectron2.data import build_detection_train_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances


register_coco_instances("my_dataset_train", {}, "dataset/annotations/instances_train.json", "dataset/images")
register_coco_instances("my_dataset_val", {}, "dataset/annotations/instances_val.json", "dataset/images")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = 'output/model_final.pt'
cfg.SOLVER.IMS_PER_BATCH = 2  
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 15000
cfg.MODEL.RETINANET.NUM_CLASSES = 15 

train_loader = build_detection_train_loader(cfg)

for batch in train_loader:
    sample_batch = batch[0]
    break

input_shape = sample_batch["image"].shape
print(f"Input tensor shape: {input_shape}")
