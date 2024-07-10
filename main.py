#Training script for the proposed model
import detectron2
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetMapper
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import torch
import os

class_names = ['car', 'bike', 'auto', 'rickshaw', 'cycle', 'bus', 'minitruck', 'truck', 'van', 'taxi', 'motorvan', 'toto', 'train', 'boat', 'cycle van']

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
cfg.MODEL.RETINANET.NUM_CLASSES = len(class_names) 

def custom_mapper(dataset_dict):
    aug = [
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
        T.RandomLighting(0.7),
        T.RandomRotation(angle=[-10, 10]),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice')
    ]
    
    mapper = DatasetMapper(is_train=True, augmentations=aug, image_format="RGB")
    return mapper(dataset_dict)

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            eval_period=200,
            model=self.model,
            data_loader=build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0]),
            cfg=self.cfg
        ))
        return hooks

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader, cfg):
        self._period = eval_period
        self._model = model
        self._data_loader = data_loader
        self.cfg = cfg

    def after_step(self):
        if self.trainer.iter % self._period == 0:
            self._do_loss_eval()

    def _do_loss_eval(self):
        self._model.eval()
        evaluator = COCOEvaluator("my_dataset_val", output_dir="./output")
        val_loader = build_detection_test_loader(self.cfg, "my_dataset_val")
        inference_on_dataset(self._model, val_loader, evaluator)
        self._model.train()

def save_model(cfg, model, output_path_pt):
    torch.save(model.state_dict(), output_path_pt)
    
trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

output_dir = cfg.OUTPUT_DIR
model_path_pt = os.path.join(output_dir, "model_final.pt")


save_model(cfg, trainer.model, model_path_pt)
