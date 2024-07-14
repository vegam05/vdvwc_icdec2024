## Introduction
The objective of this project was to develop an object detection model using the AVD dataset, which consists of 2600 images with highly imbalanced class distributions. The challenge was to accurately detect various vehicle types despite the significant class imbalance and low-quality images which were captured in diverse weather conditions.
## Methodologies
FAIR's Detectron 2's RetinaNet model was implemented to account for the class imbalances. The accuracy for object detection in diverse weather conditions can be attributed to yet another algorithm utilized in RetinaNet, i.e. Feture Pyramid Network, which creates an architecture with rich semantics at all levels by combining low-resolution semantically strong features with high-resolution semantically weak features.
## Training
The project was setup inside an Linux anaconda environment with following dependencies:
```
PyTorch==2.3.0
python==3.9.0
anaconda==24.5.0
CUDA toolkit==12.1
cuDNN==8.9.x
```
Other requirements can be installed with pip install -r requirements.txt
NOTE: If you are unable to install detectron2 for some reason, visit their official installation guide: https://detectron2.readthedocs.io/en/latest/tutorials/install.html 
Once everything is installed, run ``` main.py ``` to start training process, dataset must be in COCO format and ensure that weights and config files are initialized properly.
## Evaluation
Once the training completes, you can evaluate the mode by running ```evaluate_model.py``` to generate the normalized confusion matrix and other metrics. Ensure that the input size matches the model's input size while training, you can use ``` shape.py ``` script to find that.
## Results
After training for 41000 steps, the metrics were evaluated and stored in ``` metrices.txt ``` and ``` norm_conf_matrix.png ```.
## Misc
The ``` annotation_checker.py ``` checks if the image and their config info matches in the annotation json alongwith several other checks.
The ``` toCoco.py ``` converts the dataset in yolo format to coco taking care of the exif metadata found in jpeg images.
