# Automated Shorthand Recognition using Optimized DNNs
This repository contains the supplementary files for our research article submitted for publication. 
Please contact us before using any material in this repository.

**Authors:** Subham Sahu, Dr. Francis Jesmar P. Montalbo, Yogesh Kumar Vishwakarma, Jeevanlal Kori.

![Graphical Abstract](https://user-images.githubusercontent.com/43786036/162385236-3fee82b7-bdc4-4dfe-ac44-a4498d85219c.png)


## Requirements
- Python 3.7.7
- tensorflow == 2.7.0
- tf-keras-vis == 0.6.2
- matplotlib == 3.5.1
- opencv-python==4.5.5.62
- numpy == 1.21.5
- tqdm == 4.62.3
- scikit-learn == 1.0.2
- Pillow == 9.0.1

Use following command
```pip install -r requirements.txt```

## Directories
- *test.py :* Defined functions to test models. 
- *gui.py :* Interface Design Code.
- *models* : All trained models. [models drive link](https://drive.google.com/drive/folders/16lHHTsGacH6Ov6lxngxjHDl_pKzxfEuZ?usp=sharing)
- *ds* : Dataset  folder. [dataset drive link](https://drive.google.com/drive/folders/1uhZaogn_ksJuppiCPH_0xIGqpLywp8QT?usp=sharing)
- *utils/ :* Defined all Custom functions required for Evaluation of Approach.  
    1. *load_utils.py*: Defined functions to load trained models.
    2. *cam_utils.py*: Defined functions to produce CAMs.
    3. *ensemble.py*: Defined functions to make ensemble model.
    4. *eval_utils.py*: All other required functions are defined here.
- *notebooks/ :* Contains all Training, Evaluation and Other Notebooks.
    1. *proposed student/*: Contains proposed model training[with and withour KD] Notebooks.
    2. *teacher candidates/*: Contains teacher training Notebooks.
    3. *ROC_PR_AUC.ipynb*: Generate roc-pr plots. 

## Usage
Below command will open a Interface for testing our models.

> ```python gui.py ``` 

To run script with GPU, use below command
> ```python gui.py -g```

Two Use Cases of Interface are as follows 
- To Evaluate models [Test Accuracy, Validation Accuracy, FLOPs Count, Parameters Count].
- To Generate CAMs [Grad CAM, GradCam++, ScoreCam, Faster ScoreCam, Cameras, GuidedBP]

## Interface Guide

https://user-images.githubusercontent.com/43786036/162383581-13881c73-1a5f-4a3f-93a6-12da8ac14471.mov

**Note1:** CAMERAS will take a while to generate cams.

**Note2:** For NasNetLarge & NasNetMobile use legacy-cameras branch to produce cameras cams.
