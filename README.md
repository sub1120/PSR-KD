<!-- # Automated Shorthand Recognition using Optimized DNNs
This repository contains code used in our paper ***"Automated Shorthand Writing Recognition using an Optimized Knowledge Distilled Fine-Tuned Deep Convolutional Neural Network"*** (under review). -->


**Authors:** Subham Sahu, Dr. Francis Jesmar P. Montalbo, Yogesh Kumar Vishwakarma, Jeevanlal Kori.

This repository serves as a supplementary for a research article submitted for publication. Please contact us before using any material in this repository.

# GRAPHICAL ABSTRACT



## Requirements
- Python 3.7.7
- tensorflow 2.5.0
- tf-keras-vis 0.6.2
- keras-flops 0.1.2
- matplotlib 3.4.2
- opencv-python 4.5.2.54
- numpy 1.19.5
- tqdm 4.61.1

Use following command
```pip install -r requirements.txt```

## Directories
- *utils/ :* Defined all Custom functions required for Evaluation of Approach.  
    1. *load_utils.py*: Defined functions to load trained models.
    2. *cam_utils.py*: Defined functions to produce CAMs.
    3. *plot_utils.py*: Defined functions to draw evaluation plots.
    4. *ensemble.py*: Defined functions to make ensemble model.
    5. *eval_utils.py*: All other required functions are defined here.
- *notebooks :* Contains all Training, Evaluation and Other Notebooks.
- *test.py :* A simple script to try our all models, usage is defined below. 
- *gui.py :* GUI Script.
- *inference.py :* Calculate inference time.
- models : All trained models. [models drive link](https://drive.google.com/drive/folders/1gwIjRJpe0_Qdcl9JMxfxNz191klWTFka?usp=sharing)
- ds : Dataset  folder. [dataset drive link](https://drive.google.com/drive/folders/1HUP62HfK24o-G0KLshGGVCG44-dX2b3E?usp=sharing)

## Usage
Below command will open a Interface for testing our models.

> ```python gui.py ``` 

To run script with GPU, use below command
> ```python gui.py -g```

## Interface Guide
![Output](https://github.com/sub1120/PSR-KD/blob/master/out/GUI.PNG)

### Sample Output
|     Model Name      |   Output|
| ------------------- | --------|
| KD-EfficientNetB0 (Best Student) | ![Output](https://github.com/sub1120/PSR-KD/blob/master/out/Cams/KD-EfficientNetB0.png) |

**Note:** CAMERAS will take a while.
